# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DetectKpt loss with keypoint-aware multi-positive assignment.

Assignment, per GT armor (points TL, BL, BR, TR in 640 pixels):
1. candidates: at every level, anchors whose shape matches the GT point box (max wh ratio < anchor_t; if none
    anywhere, the best-fitting anchor), at cells that are the GT-center cell, its two YOLOv5 neighbour cells, or whose
    center lies inside the GT quadrilateral (5x5 window around the center);
2. quality q = OKS(decoded prediction, GT), alignment t = sigmoid(obj)^alpha * q^beta (task-aligned, keypoint-aware);
    the center cell with the best-fitting anchor is always kept so every GT has a positive;
3. top-k candidates per GT by t; an anchor slot claimed by two GTs goes to the higher q.
Loss: obj BCE over all anchors (positive target (1 - gr) + gr * q), and on positives 4-point L1 / sqrt(box area),
1 - OKS, CIoU of the point boxes, class / color CE, edge-parallelism + aspect-ratio consistency with the GT.
Order-uncertain GTs (flag 1) use the dihedral vertex order closest to the prediction for all point losses.
"""

import torch
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou

from ..torch_utils import de_parallel
from .metrics import align_order, kpt_area, kpt_boxes, kpt_oks, point_in_quad


def quad_features(k):
    """sin(left, right), sin(top, bottom), log(horizontal / vertical edge length) of (n, 4, 2) TL, BL, BR, TR quads."""
    left, right, top, bottom = k[:, 1] - k[:, 0], k[:, 2] - k[:, 3], k[:, 3] - k[:, 0], k[:, 2] - k[:, 1]
    ln = [v.norm(dim=1) + 1e-6 for v in (left, right, top, bottom)]
    sin_lr = (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]) / (ln[0] * ln[1])
    sin_tb = (top[:, 0] * bottom[:, 1] - top[:, 1] * bottom[:, 0]) / (ln[2] * ln[3])
    return sin_lr, sin_tb, torch.log((ln[2] + ln[3]) / (ln[0] + ln[1]))


class ComputeKptLoss:
    """Callable loss for DetectKpt training outputs: loss(p, targets) -> (total * bs, unweighted components)."""

    names = ("obj", "kpt", "oks", "box", "cls", "color", "geo")

    def __init__(self, model):
        m = de_parallel(model).model[-1]
        assert getattr(m, "decode", False), "ComputeKptLoss needs a decoded DetectKpt head (yaml kpt_decode: true)"
        self.hyp, self.device = model.hyp, m.anchors.device
        self.na, self.nl, self.nc = m.na, m.nl, m.nc
        self.stride = m.stride.to(self.device)
        self.anchors_px = m.anchors * self.stride.view(-1, 1, 1)  # (nl, na, 2) pixels
        self.balance = [4.0, 1.0, 0.4]
        r = torch.arange(-2, 3, device=self.device, dtype=torch.float32)
        self.offsets = torch.stack(torch.meshgrid(r, r, indexing="xy"), -1).view(-1, 2)  # 5x5 window (dx, dy)

    def decode(self, raw, li, a, gj, gi):
        """Same formula as DetectKpt eval: (cell + 0.5) * stride + raw * anchor -> (n, 4, 2) pixels."""
        cell = torch.stack((gi, gj), 1).float().view(-1, 1, 2)
        return (cell + 0.5) * self.stride[li].view(-1, 1, 1) + raw.view(-1, 4, 2) * self.anchors_px[li, a].view(-1, 1, 2)

    @torch.no_grad()
    def assign(self, p, targets):
        """-> positives sorted by level: li, b, a, gj, gi, gt index, quality q, GT points in matched vertex order."""
        h, nt, bs = self.hyp, targets.shape[0], p[0].shape[0]
        gk, flag = targets[:, 4:12].view(nt, 4, 2), targets[:, 3] > 0.5
        wh = (gk.max(1).values - gk.min(1).values).clamp(min=1.0)
        ctr, area = gk.mean(1), kpt_area(gk).clamp(min=1.0)
        r = wh[:, None, None] / self.anchors_px[None]  # (nt, nl, na, 2)
        fit = torch.max(r, 1 / r).max(-1).values.view(nt, -1)  # (nt, nl * na)
        best = fit.argmin(1)
        amask = (fit < h["anchor_t"]) | (torch.arange(fit.shape[1], device=self.device)[None] == best[:, None])
        amask = amask.view(nt, self.nl, self.na)

        cand = []
        off = self.offsets
        is_center = (off == 0).all(1)[None]
        for i in range(self.nl):
            s, (ny, nx) = float(self.stride[i]), p[i].shape[2:4]
            c0 = (ctr / s).floor()
            cells = c0[:, None] + off[None]  # (nt, K, 2)
            nb = torch.where(ctr / s - c0 < 0.5, -1.0, 1.0)  # YOLOv5 neighbour direction (x, y)
            is_nb = ((off[None, :, 0] == nb[:, None, 0]) & (off[None, :, 1] == 0)) | (
                (off[None, :, 1] == nb[:, None, 1]) & (off[None, :, 0] == 0)
            )
            inside = point_in_quad((cells + 0.5) * s, gk)
            inb = (cells[..., 0] >= 0) & (cells[..., 0] < nx) & (cells[..., 1] >= 0) & (cells[..., 1] < ny)
            g, a, k = (amask[:, i, :, None] & ((is_center | is_nb | inside) & inb)[:, None]).nonzero(as_tuple=True)
            forced = is_center[0, k] & (best[g] == i * self.na + a)
            cand.append((torch.full_like(g, i), g, a, cells[g, k, 1].long(), cells[g, k, 0].long(), forced))
        li, g, a, gj, gi, forced = (torch.cat(x) for x in zip(*cand))
        b = targets[g, 0].long()

        raw = torch.empty((len(g), p[0].shape[-1]), device=self.device)
        for i in range(self.nl):
            m = li == i
            raw[m] = p[i][b[m], a[m], gj[m], gi[m]].float()
        pk = self.decode(raw[:, :8], li, a, gj, gi)
        gka = align_order(pk, gk[g], flag[g])
        q = kpt_oks(pk, gka, area[g], h["oks_sigma"]).mean(1)
        t = raw[:, 8].sigmoid().pow(h["align_alpha"]) * q.pow(h["align_beta"]) + forced.float()

        order = torch.argsort(t, descending=True)
        order = order[torch.argsort(g[order], stable=True)]  # grouped by GT, t descending inside a group
        gs = g[order]
        rank = torch.arange(len(gs), device=self.device) - torch.searchsorted(gs, gs)
        sel = order[rank < h["topk"]]
        sel = sel[torch.argsort(q[sel], descending=True)]
        key = (((li[sel] * bs + b[sel]) * self.na + a[sel]) * 1024 + gj[sel]) * 1024 + gi[sel]
        _, inv = torch.unique(key, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), len(sel), device=self.device).scatter_reduce(
            0, inv, torch.arange(len(sel), device=self.device), reduce="amin"
        )
        sel = sel[first]
        sel = sel[torch.argsort(li[sel], stable=True)]
        return li[sel], b[sel], a[sel], gj[sel], gi[sel], g[sel], q[sel], gka[sel]

    def __call__(self, p, targets):
        h, bs, dev = self.hyp, p[0].shape[0], self.device
        tobj = [torch.zeros(pi.shape[:4], device=dev) for pi in p]
        zero = torch.zeros((), device=dev)
        lkpt = loks = lbox = lcls = lcolor = lgeo = zero
        if len(targets):
            li, b, a, gj, gi, g, q, gka = self.assign(p, targets)
            ps = torch.cat([p[i][b[li == i], a[li == i], gj[li == i], gi[li == i]] for i in range(self.nl)]).float()
            pk = self.decode(ps[:, :8], li, a, gj, gi)
            box_g = kpt_boxes(gka)
            scale = (box_g[:, 2:] - box_g[:, :2]).prod(1).sqrt().clamp(min=4.0)
            lkpt = ((pk - gka).abs().sum(-1).mean(-1) / scale).mean()
            loks = (1 - kpt_oks(pk, gka, kpt_area(gka).clamp(min=1.0), h["oks_sigma"])).mean()
            lbox = (1 - bbox_iou(kpt_boxes(pk), box_g, xywh=False, CIoU=True).squeeze(-1)).mean()
            tg = targets[g]
            lcls = F.cross_entropy(ps[:, 13 : 13 + self.nc], tg[:, 1].long(), label_smoothing=h["label_smoothing"])
            lcolor = F.cross_entropy(ps[:, 9:13], tg[:, 2].long(), label_smoothing=h["label_smoothing"])
            lgeo = sum((u - v).abs().mean() for u, v in zip(quad_features(pk), quad_features(gka)))
            tq = (1 - h["obj_gr"]) + h["obj_gr"] * q
            for i in range(self.nl):
                m = li == i
                tobj[i][b[m], a[m], gj[m], gi[m]] = tq[m]
        lobj = sum(w * F.binary_cross_entropy_with_logits(pi[..., 8].float(), t) for w, pi, t in zip(self.balance, p, tobj))
        parts = (lobj, lkpt, loks, lbox, lcls, lcolor, lgeo)
        total = sum(h[n] * v for n, v in zip(self.names, parts))
        return total * bs, torch.stack(parts).detach()

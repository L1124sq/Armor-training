# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Armor keypoint geometry, 27-style post-processing and validation metrics.

Metric definitions (TRAINING_REPORT.md documents them for readers):
- Armor box mAP50 / mAP50-95: YOLOv5 ap_per_class; prediction box = axis-aligned box of the 4 points; class must match.
- Pose mAP / P / R: same, but matched by OKS (sigma 0.25, area = 0.53 * GT box area, the ultralytics convention).
    Order-uncertain GTs (flag 1) are ignored for pose metrics, as are predictions whose best box match is one.
- Operating point (27 deployment): score = sigmoid(obj) >= conf, class-agnostic box NMS (IoU 0.3), then greedy
    matching by score with box IoU >= 0.3 + same class (eval_kpt_det.py rule) -> P / R / F1.
    Location-matched GTs (IoU >= 0.3, any class) give class / color top-1 and corner pixel error.
- Legacy GT-matched (old train_kpt.py口径): read the output row of the GT-center cell's best-IoU anchor, no NMS/conf.
"""

import numpy as np
import torch
import torchvision
from ultralytics.utils.metrics import box_iou

from ..metrics import ap_per_class

PERMS = (0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2), (0, 3, 2, 1), (3, 2, 1, 0), (2, 1, 0, 3), (1, 0, 3, 2)
OP_CONFS = (0.5, 0.6, 0.7, 0.8)


def kpt_boxes(k):
    """(n, 4, 2) points -> (n, 4) xyxy boxes."""
    return torch.cat((k.min(1).values, k.max(1).values), 1)


def kpt_area(k):
    """OKS scale: 0.53 * point-box area (ultralytics keypoint convention)."""
    wh = k.max(1).values - k.min(1).values
    return wh.prod(1) * 0.53


def kpt_oks(pred, gt, area, sigma=0.25):
    """Per-keypoint OKS terms (n, 4): exp(-d^2 / (8 sigma^2 area))."""
    return torch.exp(-(pred - gt).pow(2).sum(-1) / (8 * sigma**2 * area[:, None] + 1e-9))


def align_order(pred, gt, flag):
    """For order-uncertain GTs (flag True) return the dihedral vertex order closest (L1) to the prediction."""
    if not flag.any():
        return gt
    cand = gt[:, torch.tensor(PERMS, device=gt.device)]  # (n, 8, 4, 2)
    best = (cand - pred[:, None]).abs().sum((2, 3)).argmin(1)
    best = torch.where(flag, best, torch.zeros_like(best))
    return cand[torch.arange(len(gt), device=gt.device), best]


def point_in_quad(pts, quad):
    """pts (n, K, 2), convex quad (n, 4, 2) -> (n, K) bool, either vertex orientation."""
    a = quad[:, None]
    e = quad.roll(-1, 1)[:, None] - a
    v = pts[:, :, None] - a
    cross = e[..., 0] * v[..., 1] - e[..., 1] * v[..., 0]
    return (cross >= 0).all(-1) | (cross <= 0).all(-1)


def postprocess(out, nc, conf_thres=0.001, nms_iou=0.3, max_det=100, max_nms=3000):
    """27 yolov5.cpp post-processing on [bs, N, 13 + nc] -> per image (n, 12) [8 pts, score, cls, color, cls_prob]."""
    res = []
    for x in out:
        score = x[:, 8].float().sigmoid()
        i = (score > conf_thres).nonzero().squeeze(1)
        if len(i) > max_nms:
            i = i[score[i].topk(max_nms).indices]
        x, score = x[i].float(), score[i]
        k = x[:, :8].view(-1, 4, 2)
        keep = torchvision.ops.nms(kpt_boxes(k), score, nms_iou)[:max_det]
        cls_prob, cls = x[keep, 13 : 13 + nc].softmax(1).max(1)
        color = x[keep, 9:13].argmax(1)
        res.append(torch.cat((x[keep, :8], score[keep, None], cls[:, None], color[:, None], cls_prob[:, None]), 1))
    return res


def match_predictions(iou, correct_class, thresholds):
    """YOLOv5 utils.metrics.process_batch matching for any (M gts, N preds) similarity -> (N, T) bool."""
    correct = np.zeros((iou.shape[1], len(thresholds)), bool)
    for ti, t in enumerate(thresholds):
        x = torch.where((iou >= t) & correct_class)
        if x[0].shape[0]:
            m = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), ti] = True
    return correct


def greedy_match(iou, pred_cls, gt_cls, order, match_iou=0.3, need_class=True):
    """eval_kpt_det.py rule: by descending score, each prediction takes the unused GT of highest IoU; TP if that IoU
    >= match_iou (and classes equal when need_class). Returns {pred index: gt index} for TPs.
    """
    used, tp = set(), {}
    for k in order:
        best_j, best = -1, 0.0
        for j in range(iou.shape[0]):
            if j not in used and iou[j, k] > best:
                best, best_j = float(iou[j, k]), j
        if best_j >= 0 and best >= match_iou and (not need_class or pred_cls[k] == gt_cls[best_j]):
            used.add(best_j)
            tp[k] = best_j
    return tp


def legacy_index(gt_k, anchors_px, strides, na, img_size=640):
    """Old train_kpt.py assign_flat: flat output row of the GT-center cell with the best box-IoU anchor."""
    grids = [img_size // int(s) for s in strides]
    offsets = np.cumsum([0] + [na * g * g for g in grids])[:-1]
    idx = []
    for k in gt_k.cpu().numpy():
        (x0, y0), (x1, y1) = k.min(0), k.max(0)
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        best = (-1.0, 0, 0)
        for li in range(len(strides)):
            for ai in range(na):
                aw, ah = anchors_px[li][ai]
                inter = min(bw, aw) * min(bh, ah)
                v = inter / (bw * bh + aw * ah - inter)
                if v > best[0]:
                    best = (v, li, ai)
        _, li, ai = best
        s, g = int(strides[li]), grids[li]
        gi, gj = min(max(int((x0 + x1) / 2 / s), 0), g - 1), min(max(int((y0 + y1) / 2 / s), 0), g - 1)
        idx.append(offsets[li] + (ai * g + gj) * g + gi)
    return idx


class KptMetrics:
    """Accumulate per-image detections vs GTs; compute() returns a flat dict of metrics."""

    def __init__(self, nc, sigma=0.25, match_iou=0.3, op_confs=OP_CONFS):
        self.nc, self.sigma, self.match_iou, self.op_confs = nc, sigma, match_iou, op_confs
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.box, self.pose = [], []  # (correct, conf, pred_cls, target_cls)
        self.op = {c: [0, 0, 0] for c in op_confs}  # tp, fp, fn
        self.loc = {c: [] for c in op_confs}  # (cls_ok, color_ok, px_err or nan) per location-matched GT
        self.legacy = []  # (px_err, cls_ok, color_ok)
        self.per_image = []  # (fp + fn at conf 0.7, path)

    def update(self, det, gt, path="", legacy_rows=None):
        """det (n, 12) from postprocess, gt (m, 11) [cls, color, flag, 8 pts]; legacy_rows (m, 13 + nc) or None."""
        det, gt = det.cpu(), gt.cpu()
        pk, gk = det[:, :8].view(-1, 4, 2), gt[:, 3:11].view(-1, 4, 2)
        score, pcls, pcolor = det[:, 8], det[:, 9].long(), det[:, 10].long()
        gcls, gcolor, flag = gt[:, 0].long(), gt[:, 1].long(), gt[:, 2] > 0.5
        iou = box_iou(kpt_boxes(gk), kpt_boxes(pk)) if len(gt) and len(det) else torch.zeros((len(gt), len(det)))
        same = gcls[:, None] == pcls[None]

        self.box.append((match_predictions(iou, same, self.iouv), score.numpy(), pcls.numpy(), gcls.numpy()))
        # pose: unflagged GTs only; drop predictions whose best box match is a flagged GT
        keep_p = torch.ones(len(det), dtype=torch.bool)
        if len(gt) and len(det) and flag.any():
            best_iou, best_j = iou.max(0)
            keep_p = ~((best_iou >= 0.5) & flag[best_j])
        ug = ~flag
        oks = torch.zeros((int(ug.sum()), int(keep_p.sum())))
        if oks.numel():
            g, p = gk[ug], pk[keep_p]
            oks = kpt_oks(p[None], g[:, None], kpt_area(g)[:, None], self.sigma).mean(-1)
        self.pose.append(
            (match_predictions(oks, same[ug][:, keep_p], self.iouv), score[keep_p].numpy(), pcls[keep_p].numpy(), gcls[ug].numpy())
        )

        for c in self.op_confs:
            sel = (score >= c).nonzero().squeeze(1).tolist()
            order = sorted(sel, key=lambda k: -float(score[k]))
            tp = greedy_match(iou, pcls.tolist(), gcls.tolist(), order, self.match_iou)
            nfp, nfn = len(sel) - len(tp), len(gt) - len(tp)
            self.op[c][0] += len(tp)
            self.op[c][1] += nfp
            self.op[c][2] += nfn
            if c == 0.7:
                self.per_image.append((nfp + nfn, path))
            for k, j in greedy_match(iou, pcls.tolist(), gcls.tolist(), order, self.match_iou, need_class=False).items():
                err = float((pk[k] - gk[j]).norm(dim=1).mean()) if not flag[j] else float("nan")
                self.loc[c].append((int(pcls[k] == gcls[j]), int(pcolor[k] == gcolor[j]), err))

        if legacy_rows is not None:
            for r, j in zip(legacy_rows.cpu(), range(len(gt))):
                if not flag[j]:
                    err = float((r[:8].view(4, 2) - gk[j]).norm(dim=1).mean())
                    self.legacy.append((err, int(r[13 : 13 + self.nc].argmax() == gcls[j]), int(r[9:13].argmax() == gcolor[j])))

    def _ap(self, stats, prefix):
        correct, conf, pcls, tcls = (np.concatenate(x, 0) for x in zip(*stats))
        out = {f"{prefix}_mAP50": 0.0, f"{prefix}_mAP50_95": 0.0, f"{prefix}_P": 0.0, f"{prefix}_R": 0.0}
        if len(tcls) and correct.any():
            _, _, p, r, _, ap, ap_class = ap_per_class(correct, conf, pcls, tcls)
            out.update({f"{prefix}_AP50_c{c}": ap[i, 0] for i, c in enumerate(ap_class)})
            out.update({f"{prefix}_mAP50": ap[:, 0].mean(), f"{prefix}_mAP50_95": ap.mean(), f"{prefix}_P": p.mean(), f"{prefix}_R": r.mean()})
        return out

    def compute(self):
        m = {**self._ap(self.box, "box"), **self._ap(self.pose, "pose")}
        agn = [(c, s, np.zeros_like(p), np.zeros_like(t)) for c, s, p, t in self.box]  # class-agnostic armor AP
        m["armor_AP50_agnostic"] = self._ap(agn, "a")["a_mAP50"]
        for c in self.op_confs:
            tp, fp, fn = self.op[c]
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            m.update({f"P@{c}": p, f"R@{c}": r, f"F1@{c}": 2 * p * r / max(p + r, 1e-9), f"TP@{c}": tp, f"FP@{c}": fp, f"FN@{c}": fn})
            loc = np.array(self.loc[c], float).reshape(-1, 3)
            err = loc[~np.isnan(loc[:, 2]), 2]
            m.update({
                f"cls_acc@{c}": loc[:, 0].mean() if len(loc) else 0.0,
                f"color_acc@{c}": loc[:, 1].mean() if len(loc) else 0.0,
                f"px_mean@{c}": err.mean() if len(err) else float("nan"),
                f"px_median@{c}": float(np.median(err)) if len(err) else float("nan"),
                f"px_p90@{c}": float(np.percentile(err, 90)) if len(err) else float("nan"),
                f"px_max@{c}": err.max() if len(err) else float("nan"),
                f"n_loc@{c}": len(loc),
            })
        if self.legacy:
            leg = np.array(self.legacy, float)
            m.update({
                "legacy_px_mean": leg[:, 0].mean(), "legacy_px_median": float(np.median(leg[:, 0])),
                "legacy_px_max": leg[:, 0].max(), "legacy_cls_acc": leg[:, 1].mean(), "legacy_color_acc": leg[:, 2].mean(),
                "legacy_n": len(leg),
            })
        m["fitness"] = (
            0.1 * m["box_mAP50"] + 0.3 * m["box_mAP50_95"] + 0.3 * m["pose_mAP50_95"]
            + 0.1 * m["F1@0.7"] + 0.1 * m["cls_acc@0.7"] + 0.1 * m["color_acc@0.7"]
        )
        return {k: float(v) for k, v in m.items()}

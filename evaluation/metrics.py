from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist


class cal_mae:
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        self.prediction.append(np.mean(np.abs(pred - gt)))

    def show(self):
        return float(np.mean(self.prediction)) if self.prediction else 0.0


class cal_dice:
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        smooth = 1e-5
        y_true_f = gt.flatten()
        y_pred_f = pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        score = (2.0 * intersection + smooth) / (
            np.sum(y_true_f) + np.sum(y_pred_f) + smooth
        )
        self.prediction.append(score)

    def show(self):
        return float(np.mean(self.prediction)) if self.prediction else 0.0


class cal_ber:
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        binary = np.zeros_like(pred)
        binary[pred >= 0.5] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        tn = ((1 - binary) * (1 - hard_gt)).sum()
        Np = hard_gt.sum()
        Nn = (1 - hard_gt).sum()
        if Np == 0 or Nn == 0:
            ber = 0.0
        else:
            ber = 1 - (tp / (Np + 1e-8) + tn / (Nn + 1e-8)) / 2
        self.prediction.append(ber)

    def show(self):
        return float(np.mean(self.prediction)) if self.prediction else 0.0


class cal_acc:
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        binary = np.zeros_like(pred)
        binary[pred >= 0.5] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        tn = ((1 - binary) * (1 - hard_gt)).sum()
        total = hard_gt.size
        acc = (tp + tn) / (total + 1e-8)
        self.prediction.append(acc)

    def show(self):
        return float(np.mean(self.prediction)) if self.prediction else 0.0


class cal_iou:
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        smooth = 1e-5
        sample = pred.copy()
        th = min(1.0, 2 * sample.mean())
        binary = sample >= th
        target = gt > 0.5
        intersection = np.logical_and(binary, target).sum()
        union = np.logical_or(binary, target).sum()
        self.prediction.append((intersection + smooth) / (union + smooth))

    def show(self):
        return float(np.mean(self.prediction)) if self.prediction else 0.0


class cal_fm:
    def __init__(self, num, thds=255):
        self.num = max(1, num)
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.meanF = np.zeros((self.num, 1))
        self.idx = 0

    def update(self, pred, gt):
        if gt.max() == 0 or self.idx >= self.num:
            self.idx += 1
            return
        precision, recall, meanF = self.cal(pred, gt)
        self.precision[self.idx, :] = precision
        self.recall[self.idx, :] = recall
        self.meanF[self.idx, :] = meanF
        self.idx += 1

    def cal(self, pred, gt):
        th = min(1.0, 2 * pred.mean())
        binary = np.zeros_like(pred)
        binary[pred >= th] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            meanF = 0
        else:
            pre = tp / (binary.sum() + 1e-8)
            rec = tp / (hard_gt.sum() + 1e-8)
            meanF = 1.3 * pre * rec / (0.3 * pre + rec + 1e-8)
        pred_byte = np.uint8(pred * 255)
        target = pred_byte[gt > 0.5]
        nontarget = pred_byte[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(targetHist[::-1], axis=0)
        nontargetHist = np.cumsum(nontargetHist[::-1], axis=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist / (np.sum(gt) + 1e-8)
        return precision, recall, meanF

    def show(self):
        precision = self.precision[: self.idx or 1].mean(axis=0)
        recall = self.recall[: self.idx or 1].mean(axis=0)
        fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)
        fmeasure_avg = self.meanF[: self.idx or 1].mean(axis=0)
        return float(fmeasure.max()), float(fmeasure_avg[0]), precision, recall


class cal_sm:
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        score = self.cal(pred, gt > 0.5)
        self.prediction.append(score)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            return 1 - np.mean(pred)
        if y == 1:
            return np.mean(pred)
        return self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(
            pred, gt
        )

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(
            bg, np.logical_not(gt)
        )

    def s_object(self, in1, in2):
        if not np.any(in2):
            return 0.0
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (x * x + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        y, x = ndimage.center_of_mass(gt)
        y = int(round(y))
        x = int(round(x))
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)
        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]
        w1 = x * y / (area + 1e-8)
        w2 = y * (w - x) / (area + 1e-8)
        w3 = (h - y) * x / (area + 1e-8)
        w4 = (h - y) * (w - x) / (area + 1e-8)
        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        return pred[0:y, 0:x], pred[0:y, x:w], pred[y:h, 0:x], pred[y:h, x:w]

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = max(1, h * w)
        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1 + 1e-8)
        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)
        if alpha != 0:
            return alpha / (beta + 1e-8)
        if alpha == 0 and beta == 0:
            return 1.0
        return 0.0

    def show(self):
        return float(np.mean(self.prediction)) if self.prediction else 0.0


class cal_em:
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        th = min(1.0, 2 * pred.mean())
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = FM.astype(bool)
        GT = (gt > 0.5).astype(bool)
        dFM = FM.astype(float)
        if GT.sum() == 0:
            enhanced_matrix = 1.0 - dFM
        elif (~GT).sum() == 0:
            enhanced_matrix = dFM
        else:
            dGT = GT.astype(float)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        h, w = GT.shape
        score = enhanced_matrix.sum() / (h * w + 1e-8)
        self.prediction.append(score)

    def AlignmentTerm(self, dFM, dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        return (
            2
            * (align_GT * align_FM)
            / (align_GT * align_GT + align_FM * align_FM + 1e-8)
        )

    def EnhancedAlignmentTerm(self, align_Matrix):
        return np.power(align_Matrix + 1, 2) / 4

    def show(self):
        return float(np.mean(self.prediction)) if self.prediction else 0.0


class cal_wfm:
    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6
        self.scores_list = []

    def update(self, pred, gt):
        gt = gt > 0.5
        if gt.max() == 0:
            self.scores_list.append(0.0)
            return
        self.scores_list.append(self.cal(pred, gt))

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        m, n = [(ss - 1.0) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def cal(self, pred, gt):
        Dst, Idxt = bwdist(gt == 0, return_indices=True)
        E = np.abs(pred - gt)
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        MIN_E_EA = np.where(gt & (EA < E), EA, E)
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), 1.0)
        Ew = MIN_E_EA * B
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (self.eps + TPw + FPw)
        Q = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)
        return Q

    def show(self):
        return float(np.mean(self.scores_list)) if self.scores_list else 0.0


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.float32)


def _normalize_prediction(arr: np.ndarray) -> np.ndarray:
    if arr.max() == arr.min():
        return arr / 255.0
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def _normalize_gt(arr: np.ndarray) -> np.ndarray:
    if arr.max() > 0:
        arr = arr / arr.max()
    binary = np.zeros_like(arr)
    binary[arr > 0.5] = 1
    return binary


def _matching_files(pred_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    gt_files = sorted(
        [
            p
            for p in gt_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
    )
    matches: list[tuple[Path, Path]] = []
    for gt_path in gt_files:
        stem = gt_path.stem
        candidates = [
            pred_dir / f"{stem}{suffix}"
            for suffix in (gt_path.suffix, ".png", ".jpg", ".bmp")
        ]
        pred_path = next((c for c in candidates if c.exists()), None)
        if pred_path is not None:
            matches.append((pred_path, gt_path))
    return matches


def evaluate_prediction_directory(
    pred_dir: Path | str, gt_dir: Path | str
) -> dict[str, float]:
    pred_dir = Path(pred_dir).expanduser()
    gt_dir = Path(gt_dir).expanduser()
    if not pred_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"预测目录 {pred_dir} 或 GT 目录 {gt_dir} 不存在")

    pairs = _matching_files(pred_dir, gt_dir)
    if not pairs:
        raise RuntimeError(f"在 {pred_dir} 下找不到与 {gt_dir} 对应的文件")

    mae = cal_mae()
    fm = cal_fm(len(pairs))
    sm = cal_sm()
    em = cal_em()
    wfm = cal_wfm()
    m_dice = cal_dice()
    m_iou = cal_iou()
    ber = cal_ber()
    acc = cal_acc()

    for pred_path, gt_path in pairs:
        pred = _load_mask(pred_path)
        gt = _load_mask(gt_path)
        if pred.shape != gt.shape:
            pred_img = Image.fromarray(pred).resize(
                gt.shape[::-1], resample=Image.BILINEAR
            )
            pred = np.array(pred_img, dtype=np.float32)
        pred_norm = _normalize_prediction(pred)
        gt_norm = _normalize_gt(gt)
        mae.update(pred_norm, gt_norm)
        sm.update(pred_norm, gt_norm)
        em.update(pred_norm, gt_norm)
        wfm.update(pred_norm, gt_norm)
        m_dice.update(pred_norm, gt_norm)
        m_iou.update(pred_norm, gt_norm)
        ber.update(pred_norm, gt_norm)
        acc.update(pred_norm, gt_norm)
        fm.update(pred_norm, gt_norm)

    maxf, meanf, _, _ = fm.show()
    metrics = {
        "MAE": mae.show(),
        "maxF": maxf,
        "avgF": meanf,
        "wFm": wfm.show(),
        "Sm": sm.show(),
        "Em": em.show(),
        "Dice": m_dice.show(),
        "mIoU": m_iou.show(),
        "BER": ber.show(),
        "Acc": acc.show(),
    }
    return metrics

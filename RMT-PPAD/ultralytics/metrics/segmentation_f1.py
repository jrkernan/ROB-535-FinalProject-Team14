import numpy as np

class BinarySegMetrics:
    def __init__(self):
        self.cm = np.zeros((2, 2), dtype=np.int64)

    def add(self, pred, gt):
        pred = pred.astype(int).flatten()
        gt = gt.astype(int).flatten()
        for p, g in zip(pred, gt):
            self.cm[g, p] += 1

    def f1(self):
        tp = self.cm[1, 1]
        fp = self.cm[0, 1]
        fn = self.cm[1, 0]
        return (2 * tp) / (2 * tp + fp + fn + 1e-12)

    def iou(self):
        tp = self.cm[1, 1]
        fp = self.cm[0, 1]
        fn = self.cm[1, 0]
        return tp / (tp + fp + fn + 1e-12)
import torch

class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

def accuracy(logits, labels):
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return 100.0 * preds.eq(labels).sum().item() / labels.size(0)
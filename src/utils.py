import torch
import numpy as np

class Calculate(object):
    def __init__(self, name, categories, thresh, fmt=':.4f'):
        self.name = name 
        self.fmt = fmt 
        self.n_c = categories
        self.thresh = thresh
        self.mAP = 0.0
        self.mAR = 0.0
        self.Precision = [0.0] * categories
        self.Recall = [0.0] * categories
        self.TP = [0] * categories
        self.TN = [0] * categories
        self.FP = [0] * categories
        self.FN = [0] * categories
        
    def update(self, output, target):
        self.TP += np.sum( np.array(output) * np.array(target), 0 ).astype(np.int64)
        self.FP += np.sum( np.array(output) * (1 - np.array(target)), 0 ).astype(np.int64)
        self.TN += np.sum( ((np.array(output) + np.array(target)) == 0), 0 ).astype(np.int64)
        self.FN += np.sum( ((np.array(output) + (1 -np.array(target))) == 0), 0 ).astype(np.int64)
        
    def result(self):
        for i in range(self.n_c):
            if self.TP[i] + self.FP[i] == 0:
                self.Precision[i] = 0
            else:
                self.Precision[i] = self.TP[i] / (self.TP[i] + self.FP[i])

            if self.TP[i] + self.FN[i] == 0:
                self.Recall = 0
            else:
                self.Recall[i] = self.TP[i] / (self.TP[i] + self.FN[i])
        
        self.mAP = sum(self.Precision) / self.n_c
        self.mAR = sum(self.Recall) / self.n_c 
        
        if self.mAP + self.mAR == 0:
            self.mF1 = 0
        else:
            self.mF1 = 2 * self.mAP * self.mAR / (self.mAP + self.mAR)

        return self.mAP, self.mAR, self.mF1

            

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

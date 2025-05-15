import math
from urllib.request import urlretrieve
import torch
import numpy as np


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0: # 如果scores中的元素总数为0，那么直接返回0
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count == 0:
            precision_at_i = 0  # 或者使用 NaN, 取决于你的需求
        else:
            precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)


    # def evaluation(self, scores_, targets_):
    #     n, n_class = scores_.shape
    #     Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
    #     for k in range(n_class):
    #         scores = scores_[:, k]
    #         targets = targets_[:, k]
    #         targets[targets == -1] = 0
    #         Ng[k] = np.sum(targets == 1)
    #         Np[k] = np.sum(scores >= 0)
    #         Nc[k] = np.sum(targets * (scores >= 0))
    #     Np[Np == 0] = 1
    #     OP = np.sum(Nc) / np.sum(Np)
    #     OR = np.sum(Nc) / np.sum(Ng)
    #     OF1 = (2 * OP * OR) / (OP + OR)

    #     CP = np.sum(Nc / Np) / n_class
    #     CR = np.sum(Nc / Ng) / n_class
    #     CF1 = (2 * CP * CR) / (CP + CR)
    #     return OP, OR, OF1, CP, CR, CF1

# 由于类别数目的改变，导致Np[k] 即某一类的预测正样本可能为0
# 从而导致 CR 和 CF1 可能为 NaN
# 因此加入异常处理逻辑

    def evaluation(self, scores_, targets_):
        # 获取样本数量和类别数量
        n, n_class = scores_.shape
        print(f"进入evaluation，一共有{n_class}个类别")
        
        # 过滤掉完全没有样本的类别
        valid_classes_mask = np.sum(targets_, axis=0) > 0
        scores_ = scores_[:, valid_classes_mask]
        targets_ = targets_[:, valid_classes_mask]

        # 重新计算样本数量和类别数量
        n, n_class = scores_.shape
        print(f"经过过滤，有效类别数量: {n_class}")
                
        # 初始化类别相关的统计量
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        # 遍历每个类别
        # print(f'scores_ is {scores_}')
        for k in range(n_class):
            scores = scores_[:, k]  # 当前类别的预测分数
            targets = targets_[:, k]  # 当前类别的真实标签
            # print(f'in k:{k}, scores is {scores}')
            # print(f'in k:{k}, targets is {targets}')
            
            # 将目标中-1的值转换为0
            targets[targets == -1] = 0
            
            # 计算每个类别的统计量
            Ng[k] = np.sum(targets == 1)  # 正样本的数量（目标标签为1的数量）
            Np[k] = np.sum(scores >= 0)  # 预测为正样本的数量（预测分数大于等于0的数量）
            Nc[k] = np.sum(targets * (scores >= 0))  # 预测为正样本且真实标签为1的数量
            
            # 调试输出每个类别的统计信息
            # print(f"Category {k}:")
            # print(f"  Ng (Ground truth positives): {Ng[k]}")
            # print(f"  Np (Predicted positives): {Np[k]}")
            # print(f"  Nc (True positives): {Nc[k]}")
        
        # 对Np为0的情况进行处理，避免除以零
        Np[Np == 0] = 1
        
        # 计算总体性能指标
        OP = np.sum(Nc) / np.sum(Np)  # Overall precision
        OR = np.sum(Nc) / np.sum(Ng)  # Overall recall
        OF1 = (2 * OP * OR) / (OP + OR)  # Overall F1 score
        
        # 计算每个类别的性能指标，并处理Ng为0的情况
        # 对于CR，Ng为0的类别需要跳过或者用1替代，避免除以零
        valid_classes = Ng > 0  # 过滤掉Ng为0的类别
        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc[valid_classes] / np.where(Ng[valid_classes] == 0, 1, Ng[valid_classes])) / np.sum(valid_classes)  # 类别召回率
        
        # 计算加权F1得分
        CF1 = (2 * CP * CR) / (CP + CR)
        
        # 打印调试信息
        print(f"Overall Precision (OP): {OP}")
        print(f"Overall Recall (OR): {OR}")
        print(f"Overall F1 (OF1): {OF1}")
        print(f"Category Precision (CP): {CP}")
        print(f"Category Recall (CR): {CR}")
        print(f"Category F1 (CF1): {CF1}")
        
        # 返回最终的性能指标
        return OP, OR, OF1, CP, CR, CF1


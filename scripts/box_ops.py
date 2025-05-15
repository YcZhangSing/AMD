# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import math

def box_cxcywh_to_xyxy(x):  # 这个用了
    ## 输入的是归一化坐标，那么返回的也是归一化后的坐标
    # 将边界框从 中心坐标宽高格式 [cx, cy, w, h] 转换为 对角点格式 [x0, y0, x1, y1]。
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    # 将边界框从 对角点格式 [x0, y0, x1, y1] 转换为 中心坐标宽高格式 [cx, cy, w, h]。
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
# HAMMER iou
def box_iou(boxes1, boxes2, test=False):
    '''
    计算两个边界框集合的 IoU（Intersection over Union），
    并返回每个边界框对的 IoU 值和并集面积。
    '''
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    # union = area1[:, None] + area2 - inter
    union = area1 + area2 - inter

    iou = inter / union

    if test:
        zero_lines = boxes2==torch.zeros_like(boxes2)
        zero_lines_idx = torch.where(zero_lines[:,0]==True)[0]

        for idx in zero_lines_idx:
            if all(boxes1[idx,:] < 1e-4):
                iou[idx]=1

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    计算广义IoU
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    iou, union = box_iou(boxes1, boxes2)

    # lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # area = wh[:, :, 0] * wh[:, :, 1]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

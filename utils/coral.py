import torch
import torch.nn.functional as F

# CORAL 工具函数

# def coral_label_transform(y, num_classes):
#     y = y.view(-1, 1)  # (N, 1)
#     out = torch.zeros(y.size(0), num_classes - 1, device=y.device)
#     for i in range(num_classes - 1):
#         out[:, i] = (y > i).float().squeeze(1)   # !：squeeze掉多余的维度
#     return out
#
# def coral_loss(logits, y_transformed):
#     return F.binary_cross_entropy_with_logits(logits, y_transformed)

def coral_label_transform(y, num_classes):
    if y.dim() == 2:
        y = y.squeeze(1)
    y = y.long()

    batch_size = y.shape[0]
    range_tensor = torch.arange(num_classes - 1, device=y.device)
    y_expanded = y.unsqueeze(1)  # (N,1)
    targets = (y_expanded > range_tensor).float()  # (N, K-1)
    return targets


def coral_loss(logits, levels):
    loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='mean')
    return loss / levels.size(1)  # ！除以类别数
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMixConLoss(nn.Module):
    def __init__(self, temperature=0.06):
        super().__init__()
        self.temperature = temperature

    def forward(self, projection1, projection2, labels_a, labels_b, lam, index, args):
        batch_size = projection1.shape[0]
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        anchor_dot_contrast = torch.div(torch.matmul(projection2, projection1.T), self.temperature)

        mask_a = torch.eye(batch_size).cuda()
        mask_b = torch.zeros(batch_size, batch_size).cuda()
        mask_b[torch.arange(batch_size).unsqueeze(1), index.view(-1, 1)] = 1

        # 关键修复: 确保lam是正确的形状以便进行广播
        if not isinstance(lam, torch.Tensor):
            lam = torch.tensor([lam] * batch_size).cuda()
            
        # 确保lam是形状为[batch_size]的一维张量
        if len(lam.shape) == 0:  # 如果是标量张量
            lam = lam.expand(batch_size)
        elif len(lam.shape) > 1:  # 如果是多维张量
            lam = lam.view(-1)
            
        # 确保长度正确
        if len(lam) != batch_size:
            lam = lam[:batch_size]  # 截断
            if len(lam) < batch_size:
                # 扩展到正确大小
                lam = torch.cat([lam, torch.ones(batch_size - len(lam)).cuda() * lam.mean()])
        
        # 现在让lam有正确的形状用于广播
        lam_for_mask = lam.view(-1, 1)
        
        # 使用正确形状的lam进行广播操作
        mask = lam_for_mask * mask_a + (1 - lam_for_mask) * mask_b

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        exp_logits = torch.exp(logits)
        if args.negative_pair == 'diff_label':
            labels_a = labels_a.contiguous().view(-1, 1)
            logits_mask = torch.ne(labels_a, labels_a.T).cuda() + (mask_a.bool() + mask_b.bool())
            exp_logits *= logits_mask.float()

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size)

        loss = loss.mean()   
        return loss
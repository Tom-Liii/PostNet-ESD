import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import math
from helpers import one_hot_embedding, get_device
from MutualInformation import MutualInformation

def relu_evidence(y):
    return F.relu(y)

def calc_ece_softmax(softmax, label, bins=5, sample_wise=False):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    # ece = torch.zeros(1)
    batch_size = softmax.shape[0]
    ece = torch.zeros(batch_size)
    for i in range(batch_size):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = softmax_max[i].gt(bin_lower.item()) * softmax_max[i].le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0.0:
                accuracy_in_bin = correctness[i][in_bin].float().mean()
                avg_confidence_in_bin = softmax_max[i][in_bin].mean()
                # print(accuracy_in_bin, avg_confidence_in_bin)

                ece[i] += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    if sample_wise: 
        return ece
    ece = ece.mean().item()
    # print(ece)
    log_ece = -math.log(ece+1e-9)
    # print(log_ece)
    return ece

def calc_ece_evidence_u(softmax, u, label, bins=15, sample_wise=False):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    u = torch.tensor(u)
    labels = torch.tensor(label)
    softmax_max, predictions = torch.max(softmax, 1)
    # print(predictions.shape, labels.shape, softmax_max.shape)
    correctness = predictions.eq(labels)
    # correctness = correctness.unsqueeze(1)
    

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()
            

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


class DiceMetric(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceMetric, self).__init__()
        self.smooth = smooth
        self.activation = activation
        self.device = get_device()

    def dice_coef(self, softmax_pred, gt):
        """ computational formula
        """
       
        # softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        dice_ones = torch.ones(batch_size).to(self.device)
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            
            # mask = each_gt.view(batch_size,-1).sum(1) > 0
            
            # mask = mask.to(torch.int32)
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
            mask = union > 0
            mask = mask.to(torch.int32)
            dice = (2. *  intersection + 1e-5)/ (union + 1e-5)

            dice = mask * dice + (1-mask) * dice_ones
         
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)
        

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]
        
        # conver label to one-hot
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)
        
        loss = 0
        smooth = 1e-5


        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss
    
def calc_mi(outputs, labels, sample_wise=False):
    device = get_device()
    # print(device)
    _, preds = torch.max(outputs, 1)
    match = torch.eq(preds, labels)
    match = match.unsqueeze(1)
    evidence = relu_evidence(outputs)
    alpha = evidence + 1

    expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)    
    uncertainty, _ = torch.max(expected_prob, dim=1, keepdim=True)

    MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True, device=device)
    score = ( MI(match, uncertainty) + MI(uncertainty, match) ) / 2.
    if sample_wise:
        return score
    score = score.mean().item()
    return score


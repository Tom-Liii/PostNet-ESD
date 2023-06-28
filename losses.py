import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import copy

from helpers import get_device, one_hot_embedding
from metrics import calc_ece_softmax, calc_mi
from esd_dataset import target_img_size

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    alpha_ones = torch.ones(alpha.shape, dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    # print(alpha.shape)
    second_term = (
        (alpha - alpha_ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    # print(y.shape, alpha.shape)
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def policy_gradient_loss(outputs, labels, mode):
    
    device = get_device()
    batch_size = outputs.shape[0]
    num_classes = outputs.shape[1]
    y = one_hot_embedding(labels, num_classes)
    # print(y.shape)
    evidence = relu_evidence(outputs)
    alpha = evidence + 1

    expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)  
    max_prob, _ = torch.max(expected_prob, 1)
    # print(max_prob.shape)              
    ece = calc_ece_softmax(expected_prob.detach().cpu().numpy(), 
                          labels.detach().cpu().numpy(), sample_wise=True)
    mi = calc_mi(outputs, labels)
    # log_probs = torch.zeros(batch_size)
    
    # for i in range(batch_size):
    #     m = Dirichlet(alpha[i].permute(1, 2, 0).view(-1, num_classes))
    #     log_probs[i] = torch.sum(m.log_prob(expected_prob[i].permute(1, 2, 0).view(-1, num_classes))) / (target_img_size * target_img_size)
    # log_probs = log_probs.view(batch_size, -1)
    # log_probs = torch.sum(log_probs, dim=1)
    # print(log_probs, ece)
    masked_expected_prob = expected_prob * y
    # print(expected_prob.shape, labels.shape)
    
    true_class_prob = masked_expected_prob[torch.nonzero(masked_expected_prob, as_tuple=True)]
    

    # log_probs = torch.log(expected_prob[labels.unsqueeze(1)])
    log_probs = torch.log(true_class_prob)
    log_probs = torch.sum(log_probs.view(batch_size, -1), dim=1)  / (target_img_size)
    loss_ece =  log_probs * ece.to(device)
    # print(ece)
    loss_mi = -log_probs * mi
    loss_all = loss_ece + loss_mi
    if mode == 'ece':
        return loss_ece.mean()
    if mode == 'mi':
        return loss_mi.mean()
    if mode == 'ece_mi':
        return loss_all.mean()

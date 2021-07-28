"""
DAPC: https://arxiv.org/pdf/2010.03135.pdf
"""

import torch


def dapc_create_mask(lengths):
    bs = len(lengths)
    maxlen = max(lengths)
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def dapc_covariance_matrix(x, mask, T: int, reg: float = 0.0):
    B, Tmax, H = x.shape
    if torch.min(mask.sum(1)) <= T:
        raise Exception("DAPC: T is invalid for mask")

    # slice up into sliding blocks
    x_lags = x.contiguous().view([B, Tmax * H]).unfold(1, T * H, H)

    # get concatentations
    mask_lags = mask.unfold(1, T, 1).all(dim=2)
    mask_float = mask_lags.float().flatten().unsqueeze(1)
    x_lags = x_lags.reshape(-1, T * H)
    x_lags_mean = torch.sum(torch.mul(x_lags, mask_float), 0, keepdim=True) / torch.sum(
        mask_float
    )

    # subtract mean
    x_lags = x_lags - x_lags_mean

    # acutally calculate covariance matrix
    cov_est = torch.mul(x_lags, mask_float).T.matmul(x_lags) / torch.sum(mask_float)

    # regularize diagonal
    if reg > 0:
        cov_est = cov_est + reg * torch.eye(T * H, T * H, device=x.device)

    return cov_est


def dapc_predictive_information(cov_mat_two):
    split = cov_mat_two.size(0) // 2
    cov_mat_one = cov_mat_two[:split, :split]
    logdet_one = torch.logdet(cov_mat_one)
    logdet_two = torch.logdet(cov_mat_two)
    return logdet_one - 0.5 * logdet_two


# jit
dapc_cov_jit = torch.jit.script(dapc_covariance_matrix)
dapc_pi_jit = torch.jit.script(dapc_predictive_information)


def pi_loss(x, xl, T=4, div=True, reg=1e-4):
    """
    dapc mutual information loss
    """

    # covariance matrices
    mask = dapc_create_mask(xl.tolist()).to(x.device)
    cov = dapc_cov_jit(x, mask, 2 * T, reg=reg)

    # mutual information
    pi = dapc_pi_jit(cov)

    # build loss
    if div:
        loss = 1 / pi
    else:
        loss = -pi

    return loss, (mask, cov, pi)

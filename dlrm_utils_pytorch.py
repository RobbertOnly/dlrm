import torch
import time

# ### main loop ###
def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()

def dlrm_wrap(dlrm, X, lS_o, lS_i, use_gpu, device):
    if use_gpu:  # .cuda()
        return dlrm(
            X.to(device),
            [S_o.to(device) for S_o in lS_o],
            [S_i.to(device) for S_i in lS_i],
        )
    else:
        return dlrm(X, lS_o, lS_i)

def loss_fn_wrap(loss_fn, Z, T, use_gpu, device):
    if use_gpu:
        return loss_fn(Z, T.to(device))
    else:
        return loss_fn(Z, T)
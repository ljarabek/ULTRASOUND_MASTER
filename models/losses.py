import torch


def INN_loss(self, otpt, real, tightness=0.1,
             mean_loss_weight=0.1, base_loss_scalar=1):  # 3 channel output - low, mid, max;; NCHWD format
    # real format NHWD (C=1)!
    low = otpt[:, 0]
    mid = otpt[:, 1]
    high = otpt[:, 2]

    mid = torch.unsqueeze(mid, dim=1).to(device)
    zero = torch.zeros_like(real).to(device)
    # tightness = torch.tensor(tightness).to(device)
    # mean_loss_weight = torch.tensor(mean_loss_weight).to(device)
    # a = torch.max(torch.sub(real, high).to(device), other=zero).to(device)

    loss = torch.pow(torch.max(real - high, other=zero).to(device), exponent=2).to(device) + \
           torch.pow(torch.max(low - real, zero), 2)
    loss *= base_loss_scalar
    # print("Lol")
    # print(loss.mean())
    loss += tightness * (high - low)
    # print(loss.mean())
    loss += mean_loss_weight * self.loss_ce(mid.double(), real.double())
    # print(loss.mean())
    loss = loss.mean()
    return loss
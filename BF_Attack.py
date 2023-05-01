#### Load Modules ####

import os
import json
import math
import time
import numpy as np
import scipy.linalg

import torch.nn.functional as F
import loss_landscapes
from matplotlib import pyplot as plt

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('svg', 'pdf')
from matplotlib.colors import to_rgb
import matplotlib

matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns

sns.set()

## PyTorch
import torch
import torch.nn as nn

DATASET_PATH = "../Engineering Project/data"

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)


def BIM(model, x, radius=0.1, alpha=0.001, steps=8, pixelclip=(-2.6, 2.6)):
    x = x.clone().to(device)

    ### Change or move code from here on ###
    original_x = x.data
    adv_h = x.clone().detach()

    for step in range(steps):
        print("BIM Step {0}".format(step))
        adv_h.requires_grad = True
        _, wa_hat, wd_hat = model(h=adv_h)

        model.zero_grad()  # reminder why this one is necessary here?

        # Calculate loss
        # if targeted==True the labels are the targets labels else they are just the ground truth labels

        R = model.objec(h=original_x, wa=wa_hat, wd=wd_hat)

        # cost.backward(retain_graph=True)
        grad = torch.autograd.grad(R.mean(), adv_h)[0]
        # grad = torch.autograd.grad(R, adv_h)[0]

        # Grad is calculated
        delta = alpha * grad.sign()
        # Stop following gradient changes
        adv_h = adv_h.clone().detach()

        adv_h = adv_h - delta
        # print("diff: {}".format((adv_h - original_x).norm(2)))
        # Clip the change between the adverserial images and the original images to an epsilon range
        eta = torch.clamp(adv_h - original_x, min=-radius, max=radius)

        # detach to start from a fresh start images object (avoiding gradient tracking)
        # adv_x = torch.clamp(original_x + eta, min=pixelclip[0], max=pixelclip[1])
        adv_h = original_x + eta
    # ## Don't change this code:

    return adv_h, delta, wa_hat, wd_hat


##########################################################

from beam_forming import H_test, N, L, B, num_of_iter_pga, ProjGA, test_size

# ---- Classical PGA ----
# parameters defining

mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)
rates = []
# Object defining
classical_model = ProjGA(mu)
sum_rate_class, wa, wd = classical_model.forward(H_test, N, L, B, num_of_iter_pga)

wa_original, wd_original, original_h = wa.detach(), wd.detach(), H_test.detach()
print("BeamForming Rate (Un-attacked): {0}".format(
    classical_model.objec(h=original_h, wa=wa_original, wd=wd_original)[-1].mean()))
# print("BeamForming convergence: sum rate class: {0}".format(sum_rate_class[-1].mean().item()))
attack_radius = np.linspace(0.01 * 0.5, 2 * 10 * 0.01 * 2, 10)
min_dist = []
for r in attack_radius:
    print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(r))

    mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)
    bf_model = ProjGA(mu)

    adv_h, delta, wa_hat, wd_hat = BIM(bf_model, original_h, radius=r, alpha=0.01*0.5)

    adv_h = adv_h.detach()
    # plot_x_s(adv_x.numpy(), h_original.numpy(), "attacked observation", "true observation")

    # sum_rate_class, wa_attacked, wd_attacked = bf_model.forward(h=adv_h)
    # print("Attacked-BeamForming convergence: Rate: {0}".format(len(sum_rate_class)))
    print("Norm2(H_gt-H_adv): {0}".format((original_h - adv_h).norm(2)))
    attacked_rate = classical_model.objec(h=original_h, wa=wa_hat, wd=wd_hat).mean().item()
    print("BeamForming Rate (attacked): {0}".format(attacked_rate))
    rates.append(attacked_rate)

    # # ploting the results
    # plt.figure()
    # y = [r.detach().numpy() for r in (sum(sum_rate_class)/test_size)]
    # x = np.array(list(range(num_of_iter_pga))) +1
    # plt.plot(x, y, 'o')
    # plt.title(f'The Average Achievable Sum-Rate of the Test Set \n in Each Iteration of the Classical PGA')
    # plt.xlabel('Number of Iteration')
    # plt.ylabel('Achievable Rate')
    # plt.grid()
    # plt.show()

    # min_dist.append((wd_original - wa_attacked).norm(2).item())

plt.figure()
plt.plot(attack_radius, rates)
plt.xlabel('attack radius')
plt.ylabel('Rate')
plt.show()

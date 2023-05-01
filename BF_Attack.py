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


def BIM(model, x, s_gt, radius=0.1, alpha=0.001, steps=8, pixelclip=(-2.6, 2.6)):
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

        #
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


# rho_vec = np.linspace(0.1, 0.2, 40)
# for rho in rho_vec:
#     # ISTA without an attack reconstruction
#     ISTA_t_model = create_ISTA(rho=rho)
#     s_gt, err_gt = ISTA_t_model(x)
#     print("ISTA convergence: iterations: {0} | rho: {1}".format(len(err_gt), rho))
#     s_gt = s_gt.detach()
#
#     print("Performing BIM to get Adversarial Perturbation - rho: {0}".format(rho))
#     ISTA_adv_model = create_ISTA(rho=rho)
#     adv_x, delta = BIM(ISTA_adv_model, x_original, s_original)
#     adv_x = adv_x.detach()
#     plot_x_s(adv_x.numpy(), x.numpy(), "attacked observation", "true observation")
#
#     s_attacked, err_attacked = ISTA_adv_model(adv_x)
#     print("Attacked-ISTA convergence: iterations: {0}".format(len(err_attacked)))
#
#     min_dist.append((s_gt - s_attacked).norm(2).item())
#
# plt.figure()
# plt.plot(rho_vec, min_dist)
# plt.xlabel('rho')
# plt.ylabel('(S_gt-S_adv)**2')
# plt.show()

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

    adv_h, delta, wa_hat, wd_hat = BIM(bf_model, original_h, (wa_original, wd_original), radius=r, alpha=0.01*0.5)

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

########################################################################

ISTA_t_model.set_model_visualization_params()
ISTA_adv_model.set_model_visualization_params()

# (model, x,  distance=1, steps=20, normalization='model', deepcopy_model=False, adv_model=None) -> np.ndarray:
steps = 800
# I won't get the 2 plots which are the same. since the x is different
dir_one, dir_two = ISTA_t_model.get_grid_vectors(ISTA_t_model, ISTA_adv_model)

gt_line = ISTA_t_model.linear_interpolation(model_start=ISTA_t_model, model_end=ISTA_adv_model, x_sig=x,
                                            deepcopy_model=True)
adv_line = ISTA_t_model.linear_interpolation(model_start=ISTA_t_model, model_end=ISTA_adv_model, x_sig=adv_x,
                                             deepcopy_model=True)

# Plotting 1D


plt.figure()
plt.plot(np.arange(len(gt_line)), gt_line)
plt.plot(np.arange(len(adv_line)), adv_line)
plt.legend(['Ground truth loss surface', 'Adversarial loss surface'])
plt.show()

landscape_truth, landscape_adv = ISTA_t_model.random_plane(gt_model=ISTA_t_model, adv_model=ISTA_adv_model,
                                                           adv_x=adv_x, x=x,
                                                           dir_one=dir_one, dir_two=dir_two,
                                                           steps=steps)

x = np.arange(0, steps)
y = np.arange(0, steps)


def adv_f(i, j):
    return landscape_adv[i, j]


def gt_f(i, j):
    return landscape_truth[i, j]


X, Y = np.meshgrid(x, y)

Z_adv = landscape_adv
Z_gt = landscape_truth
#


# landscape_adv = ISTA_t_model.random_plane(ISTA_adv_model, x=adv_x, steps=40, dir_one=dir_one, dir_two=dir_two)
# landscape = ISTA_t_model.random_plane(ISTA_t_model, x=x, steps=40)

# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z_gt, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('surface')
# plt.show()

# Plotting 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z_adv, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('S_adv-S_gt')
ax.set_zlabel('Loss')
plt.title("Loss_adv = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
ax.view_init(30, 35)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z_gt, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('S_adv-S_gt')
ax.set_zlabel('Loss')
plt.title("Loss_gt = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
ax.view_init(30, 35)
plt.show()

# Plotting 2D


plt.figure()
cs = plt.contour(landscape_truth)
plt.clabel(cs, inline=1, fontsize=10)
plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
plt.show()

plt.figure()
cm = plt.pcolormesh(landscape_truth)
plt.colorbar(cm)
plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
plt.show()

plt.figure()
cs = plt.contour(landscape_adv)
plt.clabel(cs, inline=1, fontsize=10)
plt.colorbar(cs)
plt.title("Loss surface of adv(s) = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
plt.show()

plt.figure()
plt.pcolormesh(landscape_adv)
plt.title("Loss surface of adv(s) = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
plt.show()

pass

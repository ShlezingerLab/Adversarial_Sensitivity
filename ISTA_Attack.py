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

DATASET_PATH = "../data"

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)


def BIM(model, x, s_gt, eps=0.1, alpha=0.01, steps=5, pixelclip=(-2.6, 2.6)):
    x = x.clone().to(device)
    s_gt = s_gt.clone().to(device)

    loss = nn.MSELoss()

    ### Change or move code from here on ###
    original_x = x.data
    adv_x = x.clone().detach()

    for step in range(steps):
        print("BIM Step {0}".format(step))
        adv_x.requires_grad = True
        s_hat, errs = model(adv_x)
        model.zero_grad()

        # Calculate loss
        # if targeted==True the labels are the targets labels else they are just the ground truth labels
        cost = loss(s_gt, s_hat)

        # cost.backward(retain_graph=True)
        grad = torch.autograd.grad(cost, adv_x)[0]

        # Grad is calculated
        delta = alpha * grad.sign()

        # Stop following gradient changes
        adv_x = adv_x.clone().detach()

        adv_x = adv_x + delta

        # Clip the change between the adverserial images and the original images to an epsilon range
        eta = torch.clamp(adv_x - original_x, min=-eps, max=eps)

        # detach to start from a fresh start images object (avoiding gradient tracking)
        # adv_x = torch.clamp(original_x + eta, min=pixelclip[0], max=pixelclip[1])
        adv_x = original_x + eta

    ### Don't change this code:

    return adv_x, delta  # grad is the gradient (pertubation)


from ista import s, x, create_ISTA, plot_x_s

min_dist = []
s_original, x_original = s.detach(), x.detach()
##########################################################


rho_vec = np.linspace(0.1, 0.2, 5)
for rho in rho_vec:
    continue
    # ISTA without an attack reconstruction
    ISTA_t_model = create_ISTA(rho=rho)
    s_gt, err_gt = ISTA_t_model(x)
    print("ISTA convergence: iterations: {0} | rho: {1}".format(len(err_gt), rho))
    s_gt = s_gt.detach()

    print("Performing BIM to get Adversarial Perturbation - rho: {0}".format(rho))
    ISTA_adv_model = create_ISTA(rho=rho)
    adv_x, delta = BIM(ISTA_adv_model, x_original, s_original)
    adv_x = adv_x.detach()
    plot_x_s(adv_x.numpy(), x.numpy(), "attacked observation", "true observation")

    s_attacked, err_attacked = ISTA_adv_model(adv_x)
    print("Attacked-ISTA convergence: iterations: {0}".format(len(err_attacked)))

    min_dist.append((s_gt - s_attacked).norm(2).item())

#plt.figure()
#plt.plot(rho_vec, min_dist)
#plt.xlabel('rho')
#plt.ylabel('(S_gt-S_adv)**2')
#plt.show()

##########################################################


# ISTA without an attack reconstruction
ISTA_t_model = create_ISTA()
s_gt, err_gt = ISTA_t_model(x)
print("ISTA convergence: iterations: {0}".format(len(err_gt)))
s_gt = s_gt.detach()
eps_vec = np.linspace(0.01 * 0.5, 0.05*0.5, 2)
for e in eps_vec:
    print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(e))

    ISTA_adv_model = create_ISTA()

    adv_x, delta = BIM(ISTA_adv_model, x_original, s_original, eps=e)
    adv_x = adv_x.detach()


    s_attacked, err_attacked = ISTA_adv_model(adv_x)
    print("Attacked-ISTA convergence: iterations: {0}".format(len(err_attacked)))

    min_dist.append((s_gt - s_attacked).norm(2).item())

plt.figure()
plt.style.use('plot_style.txt')
plt.plot(adv_x.numpy(), label=r"$x_{adv}$", color='k')
plt.plot(x.numpy(), '.--', label=r"$x_{gt}$", color='r', linewidth=1)
plt.style.use('default')
plt.xlabel('Index', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend()
plt.savefig("p.pdf", bbox_inches='tight')
plt.show()

# plot_x_s(adv_x.numpy(), x.numpy(), "attacked observation", "true observation")

plt.figure()
plt.plot(eps_vec, min_dist)
plt.xlabel(r'$ { \epsilon } $')
plt.ylabel(r'${\|\| S_{gt}-S_{adv} \|\|}_2$')
plt.show()

# signal_a, signal_b, title_a='sparse signal', title_b='ISTA', errors_a=None, errors_b=None
plot_x_s(adv_x.numpy(), x.numpy(), "attacked observation", "true observation")
# plot_x_s(adv_x.detach().numpy(), s.detach().numpy(), "attacked observation", "sparse signal")

plot_x_s(signal_a=s_attacked.numpy(), title_a="Attacked ISTA reconstruction",
         signal_b=s_gt.numpy(), title_b="ISTA reconstruction",

         errors_a=err_attacked, errors_a_lbl='Attacked ISTA Convergence',
         errors_b=err_gt, errors_b_lbl="ISTA Convergence",
         print_truth_signal=True)

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
# plt.style.use('plot_style.txt')
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
ax.contour3D(X/800, Y/800, Z_adv, 50, cmap='binary')
ax.set_xlabel(r'$u_2$')
ax.set_ylabel(r'$u_1$')
ax.set_zlabel(r'Loss $\mathcal{L}$')
# plt.title("Loss_adv = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
ax.view_init(30, 35)
plt.style.use('plot_style.txt')
plt.savefig("ISTA_3D_LOSS_adv.pdf", bbox_inches='tight')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X/800, Y/800, Z_gt, 50, cmap='binary')
ax.set_xlabel(r'$u_2$')
ax.set_ylabel(r'$u_1$')
ax.set_zlabel(r'Loss $\mathcal{L}$')
# plt.title("Loss_gt = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
ax.view_init(30, 35)
plt.style.use('plot_style.txt')

plt.savefig("ISTA_3DLOSS_GT.pdf", bbox_inches='tight')
plt.show()

# Plotting 2D


plt.figure()
cs = plt.contour(landscape_truth)
plt.clabel(cs, inline=1, fontsize=10)
plt.style.use('plot_style.txt')
# plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
plt.savefig("ISTA_2D_LOSS_GT.pdf", bbox_inches='tight')

# plt.figure()
# cm = plt.pcolormesh(landscape_truth)
# plt.colorbar(cm)
# plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
# plt.show()

plt.figure()
cs = plt.contour(landscape_adv)
plt.clabel(cs, inline=1, fontsize=10)
plt.colorbar(cs)
plt.style.use('plot_style.txt')
plt.savefig("ISTA_2D_LOSS_adv.pdf", bbox_inches='tight')
# plt.title("Loss surface of adv(s) = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")

# plt.figure()
# plt.pcolormesh(landscape_adv)
# plt.title("Loss surface of adv(s) = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
# plt.show()

pass

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

import copy

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


from ista import plot_x_s
from admm import create_ADMM, s, x


min_dist = []
s_original, x_original = s.detach(), x.detach()
##########################################################


# ISTA without an attack reconstruction
ADMM_t_model = create_ADMM()
s_gt, err_gt = ADMM_t_model(x)
print("ADMM convergence: iterations: {0}".format(len(err_gt)))
s_gt = s_gt.detach()

plt.figure()
plt.subplot (2, 1, 2)
plt.plot(s[0], label = 'sparse signal', color='k')
plt.plot (s_gt ,  '.--' , label ='ADMM', color='r',linewidth=1)
plt.xlabel('Index', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend( )
plt.show()

radius_vec = np.linspace(0.01 * 0.5, 0.25, 2)
for r in radius_vec:
    print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(r))

    ADMM_adv_model = create_ADMM()

    adv_x, delta = BIM(ADMM_adv_model, x_original, s_original, eps=r)
    adv_x = adv_x.detach()

    s_attacked, err_attacked = ADMM_adv_model(adv_x)
    print("Attacked-ISTA convergence: iterations: {0}".format(len(err_attacked)))
    dist = (s_gt - s_attacked).norm(2).item()
    min_dist.append(dist)

plot_x_s(adv_x.numpy(), x.numpy(), "attacked observation", "true observation")
plt.figure()
plt.style.use('plot_style.txt')
plt.plot(radius_vec, min_dist)
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'${\|\| {s}^{\star} - {s}_{\rm adv}^{\star} \|\|}_2$')
plt.savefig("NORM2_ADMM.pdf", bbox_inches='tight')
plt.show()

#
# ISTA_min_distances = np.load('/Users/elad.sofer/src/ADVERSARIAL_SENSITIVTY/matrixes/distances_ista.npy')
# ADMM_min_distances = np.load('/Users/elad.sofer/src/ADVERSARIAL_SENSITIVTY/matrixes/distances_admm.npy')
#
# plt.figure()
# plt.style.use('plot_style.txt')
# plt.plot(radius_vec, ISTA_min_distances, '-.')
# plt.plot(radius_vec, ADMM_min_distances)
# plt.legend(['ISTA', 'ADMM'])
# plt.xlabel(r'$\epsilon$')
# plt.ylabel(r'${\|\| {s}^{\star} - {s}_{\rm adv}^{\star} \|\|}_2$')
#
# plt.savefig("f.pdf", bbox_inches='tight')
# # plt.show()

# signal_a, signal_b, title_a='sparse signal', title_b='ISTA', errors_a=None, errors_b=None
# plot_x_s(adv_x.numpy(), x.numpy(), "attacked observation", "true observation")

# plot observations
# plt.grid()
# ax = plt.subplot(2, 1, 1)
# plt.grid()
#
# plt.style.use('default')
# plt.plot(signal_a, label=r'$ x + \delta $', color='k')
# plt.xlabel('Index', fontsize=10)
# plt.ylabel('Value', fontsize=10)
# plt.legend()
#
#
# ax2 = plt.subplot(2, 1, 2)
# new_pos = ax2.get_position()
# new_pos.y0-=0.07*new_pos.y0
# new_pos.y1-=0.07*new_pos.y1
# ax2.set_position(new_pos)
# plt.style.use('default')
# plt.grid()
# plt.plot(signal_b, '.--', label=r'$x$ ', color='r', linewidth=1)
# plt.xlabel('Index', fontsize=10)
# plt.ylabel('Value', fontsize=10)
# plt.legend()
# plt.savefig("observation_combined_f.pdf", bbox_inches='tight')
# plt.show()



# plot_x_s(adv_x.detach().numpy(), s.detach().numpy(), "attacked observation", "sparse signal")

plot_x_s(signal_a=s_attacked.detach().numpy(), title_a="Attacked ADMM reconstruction",
         signal_b=s_gt.detach().numpy(), title_b="ADMM reconstruction",

         errors_a=err_attacked, errors_a_lbl='Attacked ADMM Convergence',
         errors_b=err_gt, errors_b_lbl="ADMM Convergence",
         print_truth_signal=True)

# ADMM_t_model.set_model_visualization_params()
# ADMM_adv_model.set_model_visualization_params()

# (model, x,  distance=1, steps=20, normalization='model', deepcopy_model=False, adv_model=None) -> np.ndarray:
steps = 800
# I won't get the 2 plots which are the same. since the x is different
dir_one, dir_two = ADMM_t_model.get_grid_vectors(ADMM_t_model, ADMM_adv_model)

gt_line = ADMM_t_model.linear_interpolation(model_start=ADMM_t_model, model_end=ADMM_adv_model, x_sig=x,
                                            deepcopy_model=True)
adv_line = ADMM_t_model.linear_interpolation(model_start=ADMM_t_model, model_end=ADMM_adv_model, x_sig=adv_x,
                                             deepcopy_model=True)

# Plotting 1D


plt.figure()
plt.plot(np.arange(len(gt_line)), gt_line)
plt.plot(np.arange(len(adv_line)), adv_line)
plt.legend(['Ground truth loss surface', 'Adversarial loss surface'])
plt.style.use('plot_style.txt')

plt.show()

landscape_truth, landscape_adv = ADMM_t_model.random_plane(gt_model=ADMM_t_model, adv_model=ADMM_adv_model,
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


# landscape_adv = ISTA_t_model.random_plane(ISTA_adv_model, x=adv_x, steps=40, dir_one=dir_one, dir_two=dir_two)
# landscape = ISTA_t_model.random_plane(ISTA_t_model, x=x, steps=40)

# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z_gt, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('surface')
# plt.show()

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
plt.style.use('default')
# plt.axes(projection='3d')
axs[0].view_init(30, 35)
axs[0].contour3D(X/800, Y/800, Z_adv, 50, cmap='binary')
axs[0].set_xlabel(r'$u_2$')
axs[0].set_ylabel(r'$u_1$')
axs[0].set_zlabel(r'Loss $\mathcal{L}_{adv}$')
# plt.title("Loss_adv = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
# new_post = axs[1].get_position()
new_pos = axs[1].get_position()
new_pos.x0+=0.08*new_pos.x0
new_pos.x1+=0.08*new_pos.x1
axs[1].set_position(pos=new_pos)
axs[1].contour3D(X/800, Y/800, Z_gt, 50, cmap='binary')
axs[1].set_xlabel(r'$u_2$')
axs[1].set_ylabel(r'$u_1$')
axs[1].set_zlabel(r'Loss $\mathcal{L}_{op}$')

# plt.title("Loss_gt = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
axs[1].view_init(30, 35)
# plt.style.use('plot_style.txt')
plt.savefig("ADMM_COMBINED_3D_LOSS.pdf", bbox_inches='tight')
plt.show()
# #

# plt.show()

# Plotting 2D


plt.figure()
cs = plt.contour(landscape_truth)
plt.clabel(cs, inline=1, fontsize=10)
# plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
plt.style.use('plot_style.txt')

plt.savefig("ADMM_2D_LOSS_gt.pdf", bbox_inches='tight')


# plt.figure()
# cm = plt.pcolormesh(landscape_truth)
# plt.colorbar(cm)
# # plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
# plt.show()

plt.figure()
cs = plt.contour(landscape_adv)
plt.clabel(cs, inline=1, fontsize=10)
# plt.colorbar(cs)
# plt.title("Loss surface of adv(s) = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
plt.style.use('plot_style.txt')

plt.savefig("ADMM_2D_LOSS_adv.pdf", bbox_inches='tight')

# plt.figure()
# plt.pcolormesh(landscape_adv)
# plt.title("Loss surface of adv(s) = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")
# plt.show()

pass

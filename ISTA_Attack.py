import os
import json
import math
import numpy as np
import scipy.linalg

import torch.nn.functional as F
import loss_landscapes

import seaborn as sns

sns.set()

from ista import create_ISTA
# Utilities
from utills import generate_signal, plot_conv_rec_graph, BIM, plot_3d_surface,\
    plot_2d_surface, plot_1d_surface, plot_norm_graph, plot_observations
# Configuration
from utills import sig_amount, r_step, eps_min, eps_max, loss3d_res_steps


signals = []


dist_total = np.zeros((sig_amount, r_step))
radius_vec = np.linspace(eps_min, eps_max, r_step)

for i in range(sig_amount):
    signals.append(generate_signal())
##########################################################

for sig_idx, (x_original, s_original) in enumerate(signals):
    # ISTA without an attack reconstruction
    ISTA_t_model = create_ISTA()
    s_gt, err_gt = ISTA_t_model(x_original.detach())
    print("#### ISTA signal {0} convergence: iterations: {1} ####".format(sig_idx, len(err_gt)))
    s_gt = s_gt.detach()

    for r_idx, r in enumerate(radius_vec):
        # print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(r))
        ISTA_adv_model = create_ISTA()
        adv_x, delta = BIM(ISTA_adv_model, x_original, s_original, eps=r)
        adv_x = adv_x.detach()
        s_attacked, err_attacked = ISTA_adv_model(adv_x)
        # print("Attacked-ISTA convergence: iterations: {0}".format(len(err_attacked)))

        dist_total[sig_idx, r_idx] = (s_gt - s_attacked).norm(2).item()

##########################################################
np.save('graphs/ISTA_total_norm', dist_total)
# plot_observations(adv_x, x_original, 'observation.pdf')
# plot_observations(s_attacked, s_original[0], 'observation.pdf')
plot_norm_graph(radius_vec, dist_total.mean(axis=0), fname='ISTA_norm2.pdf')

plot_conv_rec_graph(s_attacked.numpy(), s_gt.numpy(),
                    err_attacked, err_gt, fname='convergence_ISTA.pdf')

# Presenting last iteration signal loss surfaces for r=max_eps

x = x_original.detach()
ISTA_adv_model.set_model_visualization_params()
ISTA_t_model.set_model_visualization_params()

# Extract loss surface
dir_one, dir_two = ISTA_t_model.get_grid_vectors(ISTA_t_model, ISTA_adv_model)

gt_line = ISTA_t_model.linear_interpolation(model_start=ISTA_t_model, model_end=ISTA_adv_model, x_sig=x,
                                            deepcopy_model=True)
adv_line = ISTA_t_model.linear_interpolation(model_start=ISTA_t_model, model_end=ISTA_adv_model, x_sig=adv_x,
                                             deepcopy_model=True)

# Plotting 1D
plot_1d_surface(gt_line, adv_line)

Z_gt, Z_adv = ISTA_t_model.random_plane(gt_model=ISTA_t_model, adv_model=ISTA_adv_model,
                                        adv_x=adv_x, x=x,
                                        dir_one=dir_one, dir_two=dir_two,
                                        steps=loss3d_res_steps)

# Plotting 2D
plot_2d_surface(Z_gt, Z_adv)

# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# Plotting 3D
plot_3d_surface(z_adv=Z_adv, z_gt=Z_gt, steps=loss3d_res_steps, fname="ISTA_COMBINED_3D_LOSS.pdf")

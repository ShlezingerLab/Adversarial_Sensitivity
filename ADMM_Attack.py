__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import numpy as np
import matplotlib

from IPython.display import set_matplotlib_formats
import seaborn as sns

from admm import create_ADMM
from utills import generate_signal, plot_conv_rec_graph, BIM, plot_3d_surface, \
    plot_2d_surface, plot_1d_surface, plot_norm_graph, plot_observations

from utills import sig_amount, r_step, eps_min, eps_max, loss3d_res_steps


def execute():
    signals = []

    sns.set()
    set_matplotlib_formats('svg', 'pdf')
    matplotlib.rcParams['lines.linewidth'] = 2.0

    # ISTA_min_distances = np.load('stack/version1/matrices/ISTA_total_norm.npy')
    # ADMM_min_distances = np.load('stack/version1/matrices/ADMM_total_norm.npy')

    dist_total = np.zeros((sig_amount, r_step))
    radius_vec = np.linspace(eps_min, eps_max, r_step)

    for i in range(sig_amount):
        signals.append(generate_signal())

    ##########################################################

    for sig_idx, (x_original, s_original) in enumerate(signals):
        # ADMM without an attack reconstruction
        ADMM_t_model = create_ADMM()
        s_gt, err_gt = ADMM_t_model(x_original.detach())
        print("#### ADMM signal {0} convergence: iterations: {1} ####".format(sig_idx, len(err_gt)))
        s_gt = s_gt.detach()

        for r_idx, r in enumerate(radius_vec):
            # print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(r))

            ADMM_adv_model = create_ADMM()

            adv_x, delta = BIM(ADMM_adv_model, x_original, s_original, eps=r)
            adv_x = adv_x.detach()

            s_attacked, err_attacked = ADMM_adv_model(adv_x)
            # print("Attacked-ISTA convergence: iterations: {0}".format(len(err_attacked)))

            dist_total[sig_idx, r_idx] = (s_gt - s_attacked).norm(2).item()

    np.save('stack/version1/matrices/ADMM_total_norm.npy', dist_total)

    ##########################################################
    plot_norm_graph(radius_vec, dist_total.mean(axis=0), fname="NORM2_ADMM.pdf")

    # Presenting last iteration signal loss graphs for r=max_eps
    x = x_original.detach()

    plot_conv_rec_graph(s_attacked.detach().numpy(), s_gt.detach().numpy(), s_original,
                        err_attacked, err_gt,
                        fname='convergence_ADMM.pdf')

    # plot observations
    plot_observations(adv_x, x, fname="ADMM_observation.pdf")

    dir_one, dir_two = ADMM_t_model.get_grid_vectors(ADMM_t_model, ADMM_adv_model)

    # Plotting 1D
    gt_line = ADMM_t_model.linear_interpolation(model_start=ADMM_t_model, model_end=ADMM_adv_model, x_sig=x,
                                                deepcopy_model=True)
    adv_line = ADMM_t_model.linear_interpolation(model_start=ADMM_t_model, model_end=ADMM_adv_model, x_sig=adv_x,
                                                 deepcopy_model=True)

    plot_1d_surface(gt_line, adv_line, 'ADMM_1D_LOSS.pdf')

    # Plotting 2D & 3D
    Z_gt, Z_adv = ADMM_t_model.random_plane(gt_model=ADMM_t_model, adv_model=ADMM_adv_model,
                                            adv_x=adv_x, x=x, dir_one=dir_one, dir_two=dir_two, steps=loss3d_res_steps)

    np.save('stack/version1/matrices/ADMM_Z_adv.npy', Z_adv)
    np.save('stack/version1/matrices/ADMM_Z_gt.npy', Z_gt)

    plot_2d_surface(Z_gt, Z_adv, 'ADMM_2D_LOSS.pdf')

    plot_3d_surface(z_adv=Z_adv, z_gt=Z_gt, steps=loss3d_res_steps, fname='ADMM_3D_LOSS_SURFACE.pdf')


if __name__ == '__main__':
    execute()

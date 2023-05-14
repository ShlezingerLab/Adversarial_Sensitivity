__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import os

import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import orth
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('svg', 'pdf')
matplotlib.rcParams['lines.linewidth'] = 2.0

FIGURES_PATH = r'data/graphs/'
MATRICES_PATH = r'data/matrices/'

np.random.seed(0)
# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

# CONSTANTS
r_step = 40
sig_amount = 100
loss3d_res_steps = 800

eps_min, eps_max = 0.01 * 0.5, 0.05
m, n, k = 1000, 256, 5
Psi = np.eye(m)
Phi = np.random.randn(n, m)
Phi = np.transpose(orth(np.transpose(Phi)))
H = Phi
H = torch.from_numpy(H).float()


def generate_signal():
    """Generate sparse signal s and it's observation x, x=Hs+w s.t w~N(0,0.001) """
    s = np.zeros((1, m))
    index_k = np.random.choice(m, k, replace=False)
    s[:, index_k] = 0.5 * np.random.randn(k, 1).reshape([1, k])
    s = torch.from_numpy(s).float()

    # x = Hs+w s.t w~N(0,1)
    x = np.dot(H, s.T) + 0.01 * np.random.randn(n, 1)
    x = torch.from_numpy(x).float()
    return x.detach(), s.detach()


def BIM(model, x, s_gt, eps=0.1, alpha=0.01, steps=5):
    """
    Performing BIM adversarial attack for sparse-recovery casestudy.
    :param model: ADMM/ISTA object - The attacked model
    :param x: torch vector x - x=Hs+w s.t w~N(0,0.001)
    :param s_gt: torch vector s - s_gt=s^*
    :param eps: eps>0 - attack radius
    :param alpha: BIM step-size
    :param steps: how many BIM iterations to perform
    :return: adversarial x signal and the pertubation which was applied.
    """
    x = x.clone().to(device)
    s_gt = s_gt.clone().to(device)

    loss = nn.MSELoss()

    original_x = x.data
    adv_x = x.clone().detach()

    for step in range(steps):
        # print("BIM Step {0}".format(step))
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

        # Clip the change between the adversarial images and the original images to an epsilon range
        eta = torch.clamp(adv_x - original_x, min=-eps, max=eps)

        adv_x = original_x + eta

    return adv_x, delta  # grad is the gradient (perturbation)


def plot_1d_surface(gt_line, adv_line, fname):
    plt.figure()
    plt.plot(np.arange(len(gt_line)), gt_line)
    plt.plot(np.arange(len(adv_line)), adv_line)
    plt.xlabel(r'$u_1$')
    plt.legend([r'Loss $\mathcal{L}_{op}$', r'Loss $\mathcal{L}_{adv}$'])
    save_fig(fname)
    plt.show()


def plot_2d_surface(z_gt, z_adv, fname):
    plt.figure()
    cs = plt.contour(z_gt)
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(cs)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    # plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
    plt.savefig("ISTA_2D_LOSS_GT.pdf", bbox_inches='tight')

    plt.figure()
    cs = plt.contour(z_adv)
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(cs)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    save_fig(fname)
    plt.show()


def plot_3d_surface(z_adv, z_gt, steps, fname):
    x, y = np.arange(0, steps), np.arange(0, steps)
    x_vec, y_vec = np.meshgrid(x, y)

    # Plotting 3D
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    plt.style.use('default')
    # plt.axes(projection='3d')
    axs[0].view_init(30, 35)
    axs[0].contour3D(x_vec / 800, y_vec / 800, z_adv, 50, cmap='binary')
    axs[0].set_xlabel(r'$u_2$')
    axs[0].set_ylabel(r'$u_1$')
    axs[0].set_zlabel(r'Loss $\mathcal{L}_{adv}$')
    # plt.title("Loss_adv = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")

    new_pos = axs[1].get_position()
    new_pos.x0 += 0.08 * new_pos.x0
    new_pos.x1 += 0.08 * new_pos.x1
    axs[1].set_position(pos=new_pos)
    axs[1].contour3D(x_vec / 800, y_vec / 800, z_gt, 50, cmap='binary')
    axs[1].set_xlabel(r'$u_2$')
    axs[1].set_ylabel(r'$u_1$')
    axs[1].set_zlabel(r'Loss $\mathcal{L}_{op}$')
    # plt.title("Loss_gt = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
    axs[1].view_init(30, 35)
    # plt.style.use('plot_style.txt')
    save_fig(fname)
    plt.show()


def plot_conv_rec_graph(signal_a, signal_b,s_gt, errors_a, errors_b,
                        fname="convergence_ADMM.pdf"):
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    # plt.grid(color='gray')
    plt.plot(errors_a, label=r'${s}_{\rm adv}^{\star}$ convergence', linewidth=2)
    plt.plot(errors_b, '--', label=r'${s}^{\star}$ convergence', linewidth=2)
    plt.grid()
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('$\mathcal{L}$', fontsize=13)
    plt.legend()
    # plt.style.use('plot_style.txt')
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.plot(signal_a, '--*', label=r'${s}_{\rm adv}^{\star}$', color='k', linewidth=1)
    plt.plot(signal_b, '-s', label=r'${s}^{\star}$', color='r', linewidth=1)
    plt.plot(s_gt[0], '.-', label="$s$", color='g', linewidth=1)
    plt.xlabel('Index', fontsize=13)
    plt.ylabel('Value', fontsize=13)
    plt.legend()
    save_fig(fname)
    plt.show()


def plot_norm_graph(radius_vec, min_dist, fname):
    plt.figure()
    # plt.style.use('plot_style.txt')
    plt.plot(radius_vec, min_dist)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'${\|\| {s}^{\star} - {s}_{\rm adv}^{\star} \|\|}_2$')
    save_fig(fname)
    plt.show()


def plot_observations(adv_x, x, fname):
    plt.style.use('default')
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.plot(adv_x.numpy(), color='k')
    plt.legend([r"$x + \delta $"])

    axs1 = plt.subplot(2, 1, 2)
    plt.plot(x.numpy(), color='r')
    plt.grid()
    plt.legend([r"$x$"])
    new_pos = axs1.get_position()
    new_pos.y0 -= 0.08 * new_pos.y0
    new_pos.y1 -= 0.08 * new_pos.y1
    axs1.set_position(pos=new_pos)
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    save_fig(fname)
    plt.show()


def save_fig(fname):
    plt.savefig(os.path.join(FIGURES_PATH, fname), bbox_inches='tight')


if __name__ == '__main__':
    radius_vec = np.linspace(eps_min, eps_max, r_step)
    ISTA_min_distances = np.load('data/stack/version1/matrices/ISTA_total_norm.npy')
    ADMM_min_distances = np.load('data/stack/version1/matrices/ADMM_total_norm.npy')
    plt.figure()
    # plt.style.use('plot_style.txt')

    plt.plot(radius_vec, ADMM_min_distances.mean(axis=0), '.-')
    plt.plot(radius_vec, ISTA_min_distances.mean(axis=0))
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'${\|\| {s}^{\star} - {s}_{\rm adv}^{\star} \|\|}_2$')
    plt.legend(['ADMM', 'ISTA'])
    save_fig('norm2_combined.pdf')
    plt.show()

    x, s = generate_signal()
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x, label='observation')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.subplot(2, 1, 2)

    plt.plot(s[0], label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.show()

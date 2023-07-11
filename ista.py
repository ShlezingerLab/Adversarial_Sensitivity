__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import copy

import numpy as np
import torch.nn as nn
import torch
import seaborn as sns

from utills import generate_signal, plot_conv_rec_graph, BIM, plot_3d_surface, \
    plot_2d_surface, plot_1d_surface, plot_norm_graph, plot_observations
from utills import sig_amount, r_step, eps_min, eps_max, loss3d_res_steps
from visualize_model import LandscapeWrapper
from utills import m, H

sns.set()
np.random.seed(0)

# ISTA configuration
step_size = 0.1
max_iter = 10000
rho = 0.01
eps_threshold = 1e-3


# ISTA
class ISTA(nn.Module, LandscapeWrapper):
    """
       Implements the Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse signal recovery.
       Args:
           H (torch.Tensor): Sensing matrix.
           mu (float): Solver parameter for gradient descent step.
           rho (float): Regularization parameter for L1-norm penalty.
           max_iter (int): Maximum number of iterations.
           eps (float): Convergence threshold.

       Attributes:
           H (torch.Tensor): Sensing matrix.
           rho (float): Regularization parameter for L1-norm penalty.
           mu (float): Solver parameter for gradient descent step.
           max_iter (int): Maximum number of iterations.
           eps (float): Convergence threshold.
           s (torch.Tensor): Initial estimate of the sparse signal.
           model_params (nn.Parameter): Model parameters used for visualization.
       """
    def __init__(self, H, mu, rho, max_iter, eps):
        super(ISTA, self).__init__()

        # Objective parameters
        self.H = H
        self.rho = rho

        # Solver parameters
        self.mu = mu
        self.max_iter = max_iter
        self.eps = eps

        # initial estimate
        self.s = None
        self.model_params = None

    @staticmethod
    def shrinkage(x, beta):
        """
        Applies the shrinkage operator to the input tensor 'x' with a threshold of 'beta'.
        :param x: Input tensor.
        :param beta: Threshold value.
        :return: Resulting tensor after applying shrinkage.
        """
        # Shrinking towards 0 by Beta parameter.
        return torch.mul(torch.sign(x), torch.max(torch.abs(x) - beta, torch.zeros((m, 1))))

    def forward(self, x):
        """
        Performs ISTA reconstruction on the input signal 'x'.
        :param x: Input signal to reconstruct. (torch.Tensor)
        :return  torch.Tensor: Reconstructed sparse signal.
        :return list: List of recovery errors at each iteration.
        """
        self.s = torch.zeros((H.shape[1], 1))
        recovery_errors = []

        for ii in range(self.max_iter):
            s_prev = self.s
            # proximal gradient step
            temp = torch.matmul(self.H, s_prev) - x

            g_grad = s_prev - torch.mul(self.mu, torch.matmul(self.H.T, temp))
            self.s = self.shrinkage(g_grad, np.multiply(self.mu, self.rho))

            # cease if convergence achieved
            if torch.sum(torch.abs(self.s - s_prev)).item() <= self.eps:
                break

            # save recovery error
            error = self.loss_func(self.s, x)
            recovery_errors.append(error)

        return self.s, recovery_errors

    def set_model_visualization_params(self):
        """
        Sets the model parameters for visualization for the visualize_model module to operate.
        """
        self.model_params = nn.Parameter(self.s.detach(), requires_grad=False)

    def loss_func(self, s, x_sig):
        """
        Computes the loss function given the estimated sparse signal 's' and its observation 'x_sig'.
        :param s: Estimated sparse signal.
        :param x_sig: observation signal x = Hs + w, where w is a Gaussian noise.
        :return: Loss value.
        """
        return 0.5 * torch.sum((torch.matmul(self.H, s) - x_sig) ** 2).item() + self.rho * s.norm(p=1).item()

    @staticmethod
    def copy(other):
        """
        Creates a deep copy of the 'other' object.
        Args: other (ISTA): ISTA object to copy.
        Returns: ISTA: Deep copy of the 'other' object.
        """
        return copy.deepcopy(other)

    @classmethod
    def create_ISTA(cls, H=H, step_size=step_size, rho=rho, max_iter=max_iter, eps_threshold=eps_threshold):
        """
        Creates an instance of the ISTA class with the specified parameters.
        :param H: Sensing matrix.
        :param step_size: Solver parameter for gradient descent step.
        :param rho: Regularization parameter for L1-norm penalty.
        :param max_iter: Maximum number of iterations.
        :param eps_threshold: Convergence threshold.
        :return: ISTA object.
        """
        return cls(H, step_size, rho, max_iter, eps_threshold)


def execute():
    """
    Perform a series of operations on generated signals:
    1. Generate 'c' (set to 100) signals of the form x_i = Hs + w, where w follows a Gaussian distribution.
    2. Perform ISTA reconstruction on each signal x to obtain s^*.
    3. Perform BIM adversarial attack with different epsilon values to obtain x_{adv}.
    4. Perform ISTA reconstruction on each signal x_{adv} to obtain s_{adv}.
    5. Aggregate the L2 norm ||s^* - s^*_{adv}|| for each signal and epsilon value.
    6. Plot the loss surfaces in various forms (3D, 2D, 1D) and other related graphs.
    """
    signals = []
    dist_total = np.zeros((sig_amount, r_step))
    radius_vec = np.linspace(eps_min, eps_max, r_step)

    for i in range(sig_amount):
        signals.append(generate_signal())
    ##########################################################

    for sig_idx, (x_original, s_original) in enumerate(signals):
        # ISTA without an attack reconstruction
        ISTA_t_model = ISTA.create_ISTA()
        s_gt, err_gt = ISTA_t_model(x_original.detach())
        print("#### ISTA signal {0} convergence: iterations: {1} ####".format(sig_idx, len(err_gt)))
        s_gt = s_gt.detach()

        for e_idx, attack_eps in enumerate(radius_vec):
            # print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(r))
            ISTA_adv_model = ISTA.create_ISTA()
            adv_x, delta = BIM(ISTA_adv_model, x_original, s_original, eps=attack_eps)
            adv_x = adv_x.detach()
            s_attacked, err_attacked = ISTA_adv_model(adv_x)
            # print("Attacked-ISTA convergence: iterations: {0}".format(len(err_attacked)))

            dist_total[sig_idx, e_idx] = (s_gt - s_attacked).norm(2).item()

    ##########################################################
    # np.save('data/stack/version1/matrices/ISTA_total_norm.npy', dist_total)

    plot_norm_graph(radius_vec, dist_total.mean(axis=0), fname='ISTA_norm2.pdf')
    x = x_original.detach()

    plot_observations(adv_x, x, fname="ISTA_observation.pdf")
    plot_conv_rec_graph(s_attacked.detach().numpy(), s_gt.detach().numpy(), s_original,
                        err_attacked, err_gt, fname='ISTA_convergence.pdf')

    # Presenting last iteration signal loss surfaces for r=max_eps
    ISTA_adv_model.set_model_visualization_params()
    ISTA_t_model.set_model_visualization_params()

    # Extract loss surface
    dir_one, dir_two = ISTA_t_model.get_grid_vectors(ISTA_t_model, ISTA_adv_model)

    gt_line = ISTA_t_model.linear_interpolation(model_start=ISTA_t_model, model_end=ISTA_adv_model, x_sig=x,
                                                deepcopy_model=True)
    adv_line = ISTA_t_model.linear_interpolation(model_start=ISTA_t_model, model_end=ISTA_adv_model, x_sig=adv_x,
                                                 deepcopy_model=True)

    # Plotting 1D
    plot_1d_surface(gt_line, adv_line, 'ISTA_1D_LOSS.pdf')

    Z_gt, Z_adv = ISTA_t_model.random_plane(gt_model=ISTA_t_model, adv_model=ISTA_adv_model,
                                            adv_x=adv_x, x=x,
                                            dir_one=dir_one, dir_two=dir_two,
                                            steps=loss3d_res_steps)

    # np.save('data/stack/version1/matrices/ISTA_Z_adv.npy', Z_adv)
    # np.save('data/stack/version1/matrices/ISTA_Z_gt.npy', Z_gt)

    # Plotting 2D
    plot_2d_surface(Z_gt, Z_adv, 'ISTA_2D_LOSS.pdf')

    # Plotting 3D - https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    plot_3d_surface(z_adv=Z_adv, z_gt=Z_gt, steps=loss3d_res_steps, fname="ISTA_COMBINED_3D_LOSS.pdf")


if __name__ == '__main__':
    execute()
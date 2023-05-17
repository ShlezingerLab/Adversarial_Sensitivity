__author__ = 'Anonymous'

import numpy as np
import torch
import matplotlib.pyplot as plt

# official beam_forming module in - https://github.com/ortalagiv/learn-to-rapidly-optimize-hybrid-precoding
# for more info about beam-forming - https://arxiv.org/abs/2301.00369
from beam_forming import N, L, B, num_of_iter_pga, ProjGA, H_test
from utills import device


def beamforming_BIM(model, h, eps=0.1, alpha=1, steps=30):
    """
    Performs a variant of the Basic Iterative Method (BIM) attack on the beam-forming model.
    :param model: The beam-forming model.
    :param h: input matrix H.
    :param eps: The epsilon range to clip the change between adversarial and original images. (default: 0.1)
    :param alpha: The scaling factor for the gradient. (default: 1)
    :param steps: The number of BIM attack steps. (default: 30)
    :return: A tuple containing the adversarial tensor, the pertubation, delta,
             and the predicted outputs wa_hat and wd_hat.
    """

    h = h.clone().to(device)

    original_x = h.data
    adv_h = h.clone().detach()

    for step in range(steps):
        # print("BIM Step {0}".format(step))
        adv_h.requires_grad = True
        model.zero_grad()
        _, wa_hat, wd_hat = model(h=adv_h)

        R = model.objec(h=original_x, wa=wa_hat, wd=wd_hat)

        grad = torch.autograd.grad(R, adv_h)[0]

        # Grad is calculated
        delta = alpha * grad

        # Stop following gradient changes
        adv_h = adv_h.clone().detach()

        adv_h = adv_h - delta

        # Clip the change between the adverserial images and the original images to an epsilon range
        real_eta = torch.clamp((adv_h - original_x).real, min=-eps, max=eps)
        imag_eta = torch.clamp((adv_h - original_x).imag, min=-eps, max=eps)

        adv_h = original_x + torch.complex(real_eta, imag_eta)

    return adv_h, delta, wa_hat, wd_hat


##########################################################

def execute():
    """
    Executes a BIM attack on the beam-forming algorithm using different epsilon values.

    This function performs the following steps:
    1. Iterates over the dataset for each H matrix and performs the following:
       a. Creates a new instance of the ProjGA model with the given mu.
       b. Performs a BIM adversarial attack with different epsilon values on each H matrix.
       c. Computes the achievable rate.
       d. Stores the rate in the rates array at index (h_idx, e_idx).

    2. Plots a figure showing the attack radius against the mean achievable rate for all H matrices.
    """

    mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)

    classical_model = ProjGA(mu)
    sum_rate_class, wa, wd = classical_model.forward(H_test, N, L, B, num_of_iter_pga)

    wa_original, wd_original, original_h = wa.detach(), wd.detach(), H_test.detach()
    print("BeamForming Rate (Un-attacked): {0}".format(
        classical_model.objec(h=original_h, wa=wa_original, wd=wd_original).mean().norm(2).item()))

    # present noise scalar which yields 3.6 rate (benchmarking)
    noise_scalar = 0.81030
    print("BeamForming Rate (attacked via adding traditional noise with ratio) "
          "rate: {0} ratio: {1}".format(classical_model.objec(h=noise_scalar * original_h, wa=wa_original,
                                                              wd=wd_original).mean().item(), noise_scalar))

    attack_radius = np.linspace(0.002, 0.2, 20)
    mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)
    rates = np.zeros(((H_test.shape[1]), len(attack_radius)))

    for h_idx in range(H_test.shape[1]):
        original_h = H_test[:, h_idx, :, :].reshape((16, 1, 4, 12)).detach()

        for e_idx, eps in enumerate(attack_radius):
            bf_model = ProjGA(mu)

            _, _, wa_hat, wd_hat = beamforming_BIM(bf_model, original_h, eps=eps)

            attacked_rate = classical_model.objec(h=original_h, wa=wa_hat, wd=wd_hat).norm(2).item()

            rates[h_idx, e_idx] = attacked_rate

    plt.figure()
    plt.plot(attack_radius, rates.mean(axis=0))
    plt.xlabel('$\epsilon$')
    plt.ylabel('Achievable Rate')
    plt.show()


if __name__ == '__main__':
    execute()

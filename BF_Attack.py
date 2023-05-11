__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
import seaborn as sns

from utills import save_fig
from beam_forming import H_test, N, L, B, num_of_iter_pga, ProjGA

set_matplotlib_formats('svg', 'pdf')
matplotlib.rcParams['lines.linewidth'] = 2.0

sns.set()

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)


def beamforming_BIM(model, h, radius=0.1, alpha=1, steps=30):
    h = h.clone().to(device)

    original_x = h.data
    adv_h = h.clone().detach()

    for step in range(steps):
        # print("BIM Step {0}".format(step))
        adv_h.requires_grad = True
        model.zero_grad()
        _, wa_hat, wd_hat = model(h=adv_h)

        R = model.objec(h=original_x, wa=wa_hat, wd=wd_hat)
        # print("RATE: {0}".format(R.norm(2).item()))

        grad = torch.autograd.grad(R, adv_h)[0]

        # Grad is calculated
        delta = alpha * grad

        # Stop following gradient changes
        adv_h = adv_h.clone().detach()

        adv_h = adv_h - delta

        # Clip the change between the adverserial images and the original images to an epsilon range
        real_eta = torch.clamp((adv_h - original_x).real, min=-radius, max=radius)
        imag_eta = torch.clamp((adv_h - original_x).imag, min=-radius, max=radius)

        adv_h = original_x + torch.complex(real_eta, imag_eta)

    return adv_h, delta, wa_hat, wd_hat


##########################################################

def execute():
    mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)

    classical_model = ProjGA(mu)
    sum_rate_class, wa, wd = classical_model.forward(H_test, N, L, B, num_of_iter_pga)

    wa_original, wd_original, original_h = wa.detach(), wd.detach(), H_test.detach()
    print("BeamForming Rate (Un-attacked): {0}".format(
        classical_model.objec(h=original_h, wa=wa_original, wd=wd_original).mean().norm(2).item()))

    # Noise scalar which yields 3.6 Rate
    noise_scalar = 0.81030
    print("BeamForming Rate (attacked via adding traditional noise with ratio) "
          "rate: {0} ratio: {1}".format(classical_model.objec(h=noise_scalar * original_h, wa=wa_original,
                                                              wd=wd_original).mean().item(), noise_scalar))

    attack_radius = np.linspace(0.002, 0.2, 20)
    mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)
    rates = np.zeros(((H_test.shape[1]), len(attack_radius)))

    for h_idx in range(H_test.shape[1]):

        original_h = H_test[:, h_idx, :, :].reshape((16, 1, 4, 12)).detach()
        for r_idx, r in enumerate(attack_radius):
            bf_model = ProjGA(mu)
            # print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(r))
            adv_h, _, wa_hat, wd_hat = beamforming_BIM(bf_model, original_h, radius=r)
            adv_h = adv_h.detach()
            # print("Norm2(H_gt-H_adv): {0}".format((original_h - adv_h).norm(2)))
            attacked_rate = classical_model.objec(h=original_h, wa=wa_hat, wd=wd_hat).norm(2).item()
            # print("BeamForming Rate (attacked): {0}".format(attacked_rate))

            rates[h_idx, r_idx] = attacked_rate

    np.save('rates_avg.npy', rates)

    plt.figure()
    plt.plot(attack_radius, rates.mean(axis=0))
    plt.xlabel('$\epsilon$')
    plt.ylabel('${\|\|R\|\|}_{2}$')
    plt.show()


if __name__ == '__main__':
    execute()

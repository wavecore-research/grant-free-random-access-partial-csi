import matplotlib.pyplot as plt
import numpy as np

import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

import utils

np.random.seed(69)

plt.close('all')

K = 40  # Number of single-antenna users
M = 128  # Number of receive antennas
T = 40  # Preamble length
p_TX = 1
SNR_dB = 20
SNR = 10 ** (SNR_dB / 10)
beta_k = np.ones((K, 1))
eps_a = 0.25

ITER_MAX = K * 10

NUM_SIM = 10

gamma_real = np.zeros((NUM_SIM, K), dtype=complex)
gamma_prior_csi = np.zeros((NUM_SIM, K), dtype=complex)
gamma_partial_csi = np.zeros((NUM_SIM, K), dtype=complex)
gamma_no_csi = np.zeros((NUM_SIM, K), dtype=complex)

NUM_V = 100

roc_prior_csi = np.zeros((NUM_SIM, 2, NUM_V))
roc_partial_csi = np.zeros((NUM_SIM, 2, NUM_V))
roc_no_csi = np.zeros((NUM_SIM, 2, NUM_V))

## Channel generation
# lambda_k= beta_k - E(||g_k||^2/M)
lambda_k = np.zeros((K, 1)) + 0.95
lambda_compl_k = np.sqrt(1 - lambda_k ** 2)
rho = np.ones((K, 1)) * p_TX
phi = np.random.uniform(0, 2 * np.pi, size=(K, 1))

## Preamble generation and user activity
s = np.random.normal(0, 1 / np.sqrt(2), (T, K)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (T, K))
a = np.random.binomial(n=1, p=eps_a, size=(K, 1))

gamma = np.sqrt(rho) * a * np.exp(1j * phi)

## Channel generation
g = np.diag(lambda_compl_k[:, 0]) @ (
        np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (K, M)))

import tqdm
# tqdm.trange
for n_sim in range(NUM_SIM):

    gamma_real[n_sim, :] = gamma.copy().flatten()
    epsilon = np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (K, M))
    h = g + np.diag(lambda_k[:, 0]) @ epsilon

    ## Received preamble
    sigma2 = p_TX / SNR
    w = (np.random.normal(0, 1 / np.sqrt(2), (T, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (T, M))) * np.sqrt(
        sigma2)
    y = s @ np.diag(gamma[:, 0]) @ h + w

    # Estimate based on prior csi, assuming all lambdas are zeros, cf. conf. paper
    y_tilde = np.reshape(y.T, (M * T, 1))
    D = np.diag(s.reshape(K * T)) @ np.kron(np.ones((T, 1)), np.identity(K))
    Gamma = np.zeros((M * T, K), dtype=complex)
    for index_m in range(M):
        Gamma[0 + index_m * T:T + index_m * T, :] = s @ np.diag(g[:, index_m])
    gamma_hat_prior_CSI = np.linalg.inv(Gamma.conj().T @ Gamma) @ Gamma.conj().T @ y_tilde

    gamma_prior_csi[n_sim, :] = gamma_hat_prior_CSI.copy().flatten()

    C_inv_prior_CSI = np.linalg.inv(
        s @ np.diag(abs(gamma_hat_prior_CSI[:, 0]) ** 2 * lambda_k[:, 0] ** 2) @ s.T.conj() + sigma2 * np.identity(T))
    # print('Value of cost function just using only prior CSI: ' + str(
    #     utils.ML_value(gamma_hat_prior_CSI, C_inv_prior_CSI, y, s, g, M, T)))

    ## Estimator based on partial CSI and iterative ML
    # Initialization thanks to prior CSI
    gamma_hat_partial_CSI, C_inverse_partial_CSI = utils.algorithm(gamma_hat_prior_CSI, lambda_k, s, M, y, g,
                                                                   sigma2,
                                                                   T, K,
                                                                   iter_max=ITER_MAX)
    # print('Value of cost function partial CSI (prior init): ' + str(
    #     utils.ML_value(gamma_hat_partial_CSI, C_inverse_partial_CSI, y, s, g, M, T)))

    gamma_partial_csi[n_sim, :] = gamma_hat_partial_CSI.copy().flatten()

    ## Estimator based on partial CSI and iterative ML
    # Initialization without CSI
    gamma_hat_partial_CSI_0_init, C_inverse_partial_CSI_0_init = utils.algorithm(np.zeros_like(gamma), lambda_k,
                                                                                 s, M, y, g, sigma2, T, K,
                                                                                 iter_max=ITER_MAX)
    # print('Value of cost function partial CSI (0 init): ' + str(
    #     utils.ML_value(gamma_hat_partial_CSI_0_init, C_inverse_partial_CSI_0_init, y, s, g, M, T)))

    # snr_k_partial_CSI = (np.linalg.norm(g, axis=1) ** 2 + M * lambda_k[:, 0] ** 2) / sigma2

    # Estimator based on no CSI and iterative ML (as Caire)
    lambda_k = np.ones_like(lambda_k)
    gamma_hat_no_CSI, C_inverse_no_CSI = utils.algorithm(np.zeros_like(gamma), lambda_k, s, M, y, np.zeros_like(g), sigma2, T,
                                                         K, iter_max=ITER_MAX)
    gamma_no_csi[n_sim, :] = gamma_hat_no_CSI.copy().flatten()


    # snr_k_no_CSI = (np.linalg.norm(g, axis=1) ** 2 + M * lambda_k ** 2) / sigma2

    # print('Value of cost function using no CSI: ' + str(
    #     utils.ML_value(gamma_hat_no_CSI, C_inverse_no_CSI, y, s, g, M, T)))

    def roc(gamma_est):
        res = np.zeros((2, NUM_V))
        for v_i, v in enumerate(np.linspace(0, 1, NUM_V)):
            a_k_estimate = np.zeros(K, int)
            a_k_estimate[gamma_est[:, 0] > v] = 1

            x = utils.prob_false(np.abs(a[:, 0]), np.abs(a_k_estimate))
            y = utils.prob_miss(np.abs(a[:, 0]), np.abs(a_k_estimate))
            res[0, v_i] = x
            res[1, v_i] = y
        return res


    roc_prior_csi[n_sim, :, :] = roc(gamma_hat_prior_CSI)
    roc_partial_csi[n_sim, :, :] = roc(gamma_hat_partial_CSI)
    roc_no_csi[n_sim, :, :] = roc(gamma_hat_no_CSI)

# ROWS = 5
# THRESHOLD = 0.2

# plt.figure()
# plt.subplot(ROWS, 1, 1)
# plt.stem(np.abs(gamma), use_line_collection=True)
# plt.title(
#     f'gamma')
#
# ## Aggresive thresholding
# plt.subplot(ROWS, 1, 2)
# plt.stem(np.abs(gamma_hat_prior_CSI), use_line_collection=True)
# plt.title(
#     f'gamma_hat_prior_CSI')
#
# a_k_estimate = np.zeros(K, int)
# a_k_estimate[np.abs(gamma_hat_partial_CSI[:, 0]) > THRESHOLD] = 1
# plt.subplot(ROWS, 1, 3)
# plt.stem(np.abs(gamma_hat_partial_CSI), use_line_collection=True)
# plt.title(
#     f'gamma_hat_partial_CSI')
#
# a_k_estimate = np.zeros(K, int)
# a_k_estimate[np.abs(gamma_hat_partial_CSI_0_init[:, 0]) > THRESHOLD] = 1
# plt.subplot(ROWS, 1, 4)
# plt.stem(np.abs(gamma_hat_partial_CSI_0_init), use_line_collection=True)
# plt.title(
#     f'gamma_hat_partial_CSI_0_init')
#
# a_k_estimate = np.zeros(K, int)
# a_k_estimate[np.abs(gamma_hat_no_CSI[:, 0]) > THRESHOLD] = 1
# plt.subplot(ROWS, 1, 5)
# plt.stem(np.abs(gamma_hat_no_CSI), use_line_collection=True)
# plt.title(
#     f'gamma_hat_no_CSI')
#
# print(f'MSE just using prior CSI: \t{utils.MSE(gamma, gamma_hat_prior_CSI):0.2f} dB')
# print(f'MSE using partial CSI:  \t{utils.MSE(gamma, gamma_hat_partial_CSI):0.2f} dB')
# print(f'MSE using no CSI: \t\t\t{utils.MSE(gamma, gamma_hat_no_CSI):0.2f} dB')
#
# plt.tight_layout()
# plt.show()


ROWS = 5
THRESHOLD = 0.2

plt.figure()
_roc = np.average(roc_prior_csi, axis=0)
plt.plot(_roc[0], _roc[1], label="Prior CSI")
_roc = np.average(roc_partial_csi, axis=0)
plt.plot(_roc[0], _roc[1], label="Partial CSI")
_roc = np.average(roc_no_csi, axis=0)
plt.plot(_roc[0], _roc[1], label="No CSI")

plt.yscale('log')
plt.xscale('log')

plt.xlabel("Probability of False Alarm")
plt.ylabel("Probability of Miss Detection")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
_roc = np.average(roc_prior_csi, axis=0)
plt.plot(_roc[0] * 100, _roc[1] * 100, label="Prior CSI")
_roc = np.average(roc_partial_csi, axis=0)
plt.plot(_roc[0] * 100, _roc[1] * 100, label="Partial CSI")
_roc = np.average(roc_no_csi, axis=0)
plt.plot(_roc[0] * 100, _roc[1] * 100, label="No CSI")

# plt.yscale('log')
# plt.xscale('log')

plt.xlabel("Probability of False Alarm (%)")
plt.ylabel("Probability of Miss Detection (%)")
plt.legend()
plt.tight_layout()
plt.show()

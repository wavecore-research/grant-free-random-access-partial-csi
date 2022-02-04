import matplotlib.pyplot as plt
import numpy as np

import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

import utils

np.random.seed(0)

plt.close('all')

#K = 40  # Number of single-antenna users
M = 64  # Number of receive antennas
p_TX = 1

# beta_k = np.ones((K, 1))
eps_a = 0.25



NUM_MONTE_SIM = 20
NUM_LAMBDA = 10
NUM_SNR = 10
NUM_T = 10

lambdas = np.linspace(0.1, 0.95, num=NUM_LAMBDA)
snrs_dB = np.linspace(20, -20, num=NUM_SNR)
snrs = 10 ** (snrs_dB / 10)
preamble_lengths = np.arange(4, 100, step=100 // NUM_T, dtype=int)
antennas = [16, 32, 64, 128]
users = [40, 60, 100, 120, 200, 500] # Number of single-antenna users

NUM_ANT = len(antennas)
NUM_K = len(users)

# NUM_SIM = NUM_MONTE_SIM * NUM_LAMBDA * NUM_SNR

# SHAPE_GAMMA = (NUM_MONTE_SIM, NUM_LAMBDA, NUM_SNR, NUM_T, K)
SHAPE_MSE = (NUM_MONTE_SIM, NUM_LAMBDA, NUM_SNR, NUM_T, NUM_ANT, NUM_K)

# gamma_real = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_prior_csi = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_partial_csi = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_no_csi = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_genie_csi = np.zeros(SHAPE_GAMMA, dtype=complex)

MSE_prior_csi = np.zeros(SHAPE_MSE, dtype=float)
MSE_partial_csi = np.zeros(SHAPE_MSE, dtype=float)
MSE_no_csi = np.zeros(SHAPE_MSE, dtype=float)
# MSE_genie_csi = np.zeros(SHAPE_MSE, dtype=float)

import tqdm
pbar = tqdm.tqdm(total=NUM_MONTE_SIM*NUM_LAMBDA*NUM_SNR*NUM_T*NUM_ANT*NUM_K)

for n_sim in range(NUM_MONTE_SIM):
    for i_K, K in enumerate(users):
        ITER_MAX = K * 10
        rho = np.ones((K, 1)) * p_TX
        phi = np.random.uniform(0, 2 * np.pi, size=(K, 1))
        a = np.random.binomial(n=1, p=eps_a, size=(K, 1))
        gamma = np.sqrt(rho) * a * np.exp(1j * phi)

        for i_T, T in enumerate(preamble_lengths):

            ## Preamble generation and user activity
            s = np.random.normal(0, 1 / np.sqrt(2), (T, K)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (T, K))

            for i_lmbda, lambda_corr in enumerate(lambdas):

                ## Channel generation
                # lambda_k= beta_k - E(||g_k||^2/M)
                lambda_k = np.zeros((K, 1)) + lambda_corr
                lambda_compl_k = np.sqrt(1 - lambda_k ** 2)

                for i_M, M in enumerate(antennas):

                    ## Channel generation
                    g = np.diag(lambda_compl_k[:, 0]) @ (
                            np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2),
                                                                                                (K, M)))

                    for i_snr, snr in enumerate(snrs):

                        # gamma_real[n_sim, i_lmbda, i_snr, :] = gamma.copy().flatten()

                        epsilon = np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2),
                                                                                                      (K, M))
                        h = g + np.diag(lambda_k[:, 0]) @ epsilon

                        ## Received preamble
                        sigma2 = p_TX / snr
                        w = (np.random.normal(0, 1 / np.sqrt(2), (T, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2),
                                                                                                 (T, M))) * np.sqrt(
                            sigma2)
                        y = s @ np.diag(gamma[:, 0]) @ h + w

                        # Estimate based on prior csi, assuming all lambdas are zeros, cf. conf. paper
                        y_tilde = np.reshape(y.T, (M * T, 1))
                        D = np.diag(s.reshape(K * T)) @ np.kron(np.ones((T, 1)), np.identity(K))
                        Gamma = np.zeros((M * T, K), dtype=complex)
                        for index_m in range(M):
                            Gamma[0 + index_m * T:T + index_m * T, :] = s @ np.diag(g[:, index_m])
                        gamma_hat_prior_CSI = np.linalg.inv(Gamma.conj().T @ Gamma) @ Gamma.conj().T @ y_tilde

                        # gamma_prior_csi[n_sim, i_lmbda, i_snr, i_T, :] = gamma_hat_prior_CSI.copy()[:, 0]

                        C_inv_prior_CSI = np.linalg.inv(
                            s @ np.diag(
                                np.abs(gamma_hat_prior_CSI[:, 0]) ** 2 * lambda_k[:,
                                                                         0] ** 2) @ s.T.conj() + sigma2 * np.identity(
                                T))

                        MSE_prior_csi[n_sim, i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma, gamma_hat_prior_CSI[:,
                                                                                               0])  # utils.ML_value(gamma_prior_csi[n_sim, :], C_inv_prior_CSI, y, s, g, M, T)

                        ## Estimator based on partial CSI and iterative ML
                        # Initialization thanks to prior CSI
                        gamma_hat_partial_CSI, C_inverse_partial_CSI = utils.algorithm(gamma_hat_prior_CSI.copy(),
                                                                                       lambda_k, s,
                                                                                       M,
                                                                                       y, g,
                                                                                       sigma2,
                                                                                       T, K,
                                                                                       iter_max=ITER_MAX)
                        # gamma_partial_csi[n_sim, i_lmbda, i_snr, i_T, :] = gamma_hat_partial_CSI.copy()

                        MSE_partial_csi[n_sim, i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma,
                                                                                          gamma_hat_partial_CSI)  # utils.ML_value(gamma_partial_csi[n_sim, :], C_inverse_partial_CSI, y, s, g, M, T)

                        # gamma_hat_genie_CSI, C_inverse_genie_CSI = utils.algorithm(gamma, lambda_k, s, M, y, g,
                        #                                                            sigma2,
                        #                                                            T, K,
                        #                                                            iter_max=ITER_MAX)
                        # gamma_genie_csi[n_sim, :] = gamma_hat_genie_CSI.copy()
                        # MSE_genie_csi[n_sim] = utils.MSE(gamma_real[n_sim, :], gamma_genie_csi[n_sim,
                        #                                                        :])  # utils.ML_value(gamma_genie_csi[n_sim, :], C_inverse_genie_CSI, y, s, g, M, T)

                        ## Estimator based on partial CSI and iterative ML
                        # Initialization without CSI
                        # gamma_hat_partial_CSI_0_init, C_inverse_partial_CSI_0_init = utils.algorithm(np.zeros_like(gamma), lambda_k,
                        #                                                                              s, M, y, g, sigma2, T, K,
                        #                                                                              iter_max=ITER_MAX)

                        # utils.ML_value(gamma_hat_partial_CSI_0_init, C_inverse_partial_CSI_0_init, y, s, g, M, T)))

                        # snr_k_partial_CSI = (np.linalg.norm(g, axis=1) ** 2 + M * lambda_k[:, 0] ** 2) / sigma2

                        # Estimator based on no CSI and iterative ML (as Caire)
                        gamma_hat_no_CSI, C_inverse_no_CSI = utils.algorithm(np.zeros_like(gamma),
                                                                             np.ones_like(lambda_k), s, M,
                                                                             y,
                                                                             np.zeros_like(g),
                                                                             sigma2, T,
                                                                             K, iter_max=ITER_MAX)
                        # gamma_no_csi[n_sim, i_lmbda, i_snr, i_T:] = gamma_hat_no_CSI.copy()

                        # snr_k_no_CSI = (np.linalg.norm(g, axis=1) ** 2 + M * lambda_k ** 2) / sigma2

                        MSE_no_csi[n_sim, i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma,
                                                                                     gamma_hat_no_CSI)  # utils.ML_value(gamma_no_csi[n_sim, :], C_inverse_no_CSI, y, s, g, M, T)
                        pbar.update()
"""
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(snrs_dB, lambdas)
surf = ax.plot_surface(X, Y, 10 * np.log10(np.average(MSE_prior_csi, axis=0)), label="prior CSI", alpha=0.4)
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d
surf = ax.plot_surface(X, Y, 10 * np.log10(np.average(MSE_partial_csi, axis=0)), label="partial CSI", alpha=0.4)
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d
surf = ax.plot_surface(X, Y, 10 * np.log10(np.average(MSE_no_csi, axis=0)), label="no CSI", alpha=0.4)
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('$\lambda_k$')
ax.set_zlabel('MSE')
plt.legend()
plt.tight_layout()
plt.show()

import plotly.graph_objects as go

colorscale_1 = [[0, "rgb(107,184,255)"], [1, "rgb(0,90,124)"]]
colorscale_2 = [[0, "rgb(255,107,184)"], [1, "rgb(128,0,64)"]]
colorscale_3 = [[0, "rgb(107,255,184)"], [1, "rgb(0,124,90)"]]

fig = go.Figure(data=[
    go.Surface(name="Prior CSI", x=snrs_dB, y=lambdas, z=10 * np.log10(np.average(MSE_prior_csi, axis=0)), opacity=0.5,
               colorscale=colorscale_1),
    go.Surface(name="Partial CSI", x=snrs_dB, y=lambdas, z=10 * np.log10(np.average(MSE_partial_csi, axis=0)),
               showscale=False, opacity=0.5, colorscale=colorscale_2),
    go.Surface(name="No CSI", x=snrs_dB, y=lambdas, z=10 * np.log10(np.average(MSE_no_csi, axis=0)), showscale=False,
               opacity=0.5, colorscale=colorscale_3),
])
fig.update_layout(scene=dict(
    xaxis_title='SNR (dB)',
    yaxis_title='lambda_k',
    zaxis_title='MSE (dB)'))
fig.show()

fig = plt.figure()
plt.plot(snrs_dB, 10 * np.log10(np.average(MSE_prior_csi[:, 0, :], axis=0)),
         label=f"prior CSI $\lambda_k$={lambdas[0]:0.2f}")
plt.plot(snrs_dB, 10 * np.log10(np.average(MSE_prior_csi[:, -1, :], axis=0)),
         label=f"prior CSI $\lambda_k$={lambdas[-1]:0.2f}",
         color=plt.gca().lines[-1].get_color(), ls="--")

plt.plot(snrs_dB, 10 * np.log10(np.average(MSE_partial_csi[:, 0, :], axis=0)), label="partial CSI")
plt.plot(snrs_dB, 10 * np.log10(np.average(MSE_partial_csi[:, -1, :], axis=0)),
         color=plt.gca().lines[-1].get_color(), ls="--")
plt.plot(snrs_dB, 10 * np.log10(np.average(MSE_no_csi[:, 0, :], axis=0)), label="no CSI")
plt.plot(snrs_dB, 10 * np.log10(np.average(MSE_no_csi[:, -1, :], axis=0)),
         color=plt.gca().lines[-1].get_color(), ls="--")

plt.xlabel("SNR (dB)")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(lambdas, 10 * np.log10(np.average(MSE_prior_csi[:, :, 0], axis=0)), label=f"prior CSI snr ={snrs_dB[0]} dB")
plt.plot(lambdas, 10 * np.log10(np.average(MSE_prior_csi[:, :, -1], axis=0)), label=f"prior CSI snr ={snrs_dB[-1]} dB",
         color=plt.gca().lines[-1].get_color(), ls="--")

plt.plot(lambdas, 10 * np.log10(np.average(MSE_partial_csi[:, :, 0], axis=0)),
         label=f"partial CSI snr ={snrs_dB[0]} dB")
plt.plot(lambdas, 10 * np.log10(np.average(MSE_partial_csi[:, :, -1], axis=0)),
         label=f"partial CSI snr ={snrs_dB[-1]} dB",
         color=plt.gca().lines[-1].get_color(), ls="--")
plt.plot(lambdas, 10 * np.log10(np.average(MSE_no_csi[:, :, 0], axis=0)), label=f"no CSI snr ={snrs_dB[0]} dB")
plt.plot(lambdas, 10 * np.log10(np.average(MSE_no_csi[:, :, -1], axis=0)), label=f"no CSI snr ={snrs_dB[-1]} dB",
         color=plt.gca().lines[-1].get_color(), ls="--")

plt.xlabel("$\lambda_k$")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

# TODO sum log or sum 10**log/10?
# print(f'MSE just using prior CSI: \t\t{np.average(MSE_prior_csi):0.2f} dB')
# print(f'MSE using partial CSI:  \t\t{np.average(MSE_partial_csi):0.2f} dB')
# print(f'MSE using genie-aided CSI:  \t{np.average(MSE_genie_csi):0.2f} dB')
# print(f'MSE using no CSI: \t\t\t\t{np.average(MSE_no_csi):0.2f} dB')
#
# ROWS = 5
# THRESHOLD = 0.2
#
# plt.figure()
# _roc = np.average(roc_prior_csi, axis=0)
# plt.plot(_roc[0], _roc[1], label="Prior CSI")
# _roc = np.average(roc_partial_csi, axis=0)
# plt.plot(_roc[0], _roc[1], label="Partial CSI")
# _roc = np.average(roc_no_csi, axis=0)
# plt.plot(_roc[0], _roc[1], label="No CSI")
#
# plt.yscale('log')
# plt.xscale('log')
#
# plt.xlabel("Probability of False Alarm")
# plt.ylabel("Probability of Miss Detection")
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# plt.figure()
# _roc = np.average(roc_prior_csi, axis=0)
# plt.plot(_roc[0] * 100, _roc[1] * 100, label="Prior CSI")
# _roc = np.average(roc_partial_csi, axis=0)
# plt.plot(_roc[0] * 100, _roc[1] * 100, label="Partial CSI")
# _roc = np.average(roc_no_csi, axis=0)
# plt.plot(_roc[0] * 100, _roc[1] * 100, label="No CSI")
# _roc = np.average(roc_genie_csi, axis=0)
# plt.plot(_roc[0] * 100, _roc[1] * 100, label="Genie-aided CSI")
#
# # plt.yscale('log')
# # plt.xscale('log')
#
# plt.xlabel("Probability of False Alarm (%)")
# plt.ylabel("Probability of Miss Detection (%)")
# plt.legend()
# plt.tight_layout()
# plt.show()
"""
pbar.close()
np.savez("data", MSE_prior_csi=MSE_prior_csi, MSE_partial_csi=MSE_partial_csi, MSE_no_csi=MSE_no_csi)

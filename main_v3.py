import os
import warnings

import matplotlib

matplotlib.use('Qt5Agg')

import utils
import tqdm
import matplotlib.pyplot as plt

import numpy as np
from numba.core.errors import NumbaPerformanceWarning

# import jax.numpy as np

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

np.random.seed(0)

plt.close('all')

# K = 40  # Number of single-antenna users
# M = 64  # Number of receive antennas
p_TX = 1

# beta_k = np.ones((K, 1))
eps_a = 0.1

NUM_MONTE_SIM = 2
NUM_NOISE_REALIZATIONS = 50
NUM_LAMBDA = 1
NUM_SNR = 10
NUM_T = 4  # number of diff preambles per run 10->40

NUM_V = 200

lambdas = np.linspace(0.1, 0.95, num=NUM_LAMBDA)
preamble_lengths = np.arange(10, 40, step=(40 - 10) // NUM_T, dtype=int)

snrs_dB = np.linspace(20, -20, num=NUM_SNR)
snrs = 10 ** (np.asarray(snrs_dB) / 10)
antennas = [32] #, 64, 128]

users = [10 , 60] #, 60, 100, 120, 200, 500]  # Number of single-antenna users

# Td = 40  # 40 payload symbols

NUM_ANT = len(antennas)
NUM_K = len(users)

# NUM_SIM = NUM_MONTE_SIM * NUM_LAMBDA * NUM_SNR

# SHAPE_GAMMA = (NUM_MONTE_SIM, NUM_LAMBDA, NUM_SNR, NUM_T, K)
SHAPE_MSE = (NUM_LAMBDA, NUM_SNR, NUM_T, NUM_ANT, NUM_K)
SHAPE_PROB = (NUM_LAMBDA, NUM_SNR, NUM_T, NUM_ANT, NUM_K, NUM_V)

# gamma_real = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_prior_csi = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_partial_csi = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_no_csi = np.zeros(SHAPE_GAMMA, dtype=complex)
# gamma_genie_csi = np.zeros(SHAPE_GAMMA, dtype=complex)

MSE_prior_csi = np.zeros(SHAPE_MSE, dtype=float)
MSE_partial_csi = np.zeros(SHAPE_MSE, dtype=float)
MSE_partial_csi_ZF = np.zeros(SHAPE_MSE, dtype=float)
MSE_no_csi = np.zeros(SHAPE_MSE, dtype=float)
# MSE_genie_csi = np.zeros(SHAPE_MSE, dtype=float)


pa_prior_csi = np.zeros(SHAPE_PROB, dtype=float)
md_prior_csi = np.zeros(SHAPE_PROB, dtype=float)

pa_no_csi = np.zeros(SHAPE_PROB, dtype=float)
md_no_csi = np.zeros(SHAPE_PROB, dtype=float)

pa_partial_csi = np.zeros(SHAPE_PROB, dtype=float)
md_partial_csi = np.zeros(SHAPE_PROB, dtype=float)

pa_partial_csi_ZF = np.zeros(SHAPE_PROB, dtype=float)
md_partial_csi_ZF = np.zeros(SHAPE_PROB, dtype=float)

pbar = tqdm.tqdm(total=NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS * NUM_LAMBDA * NUM_SNR * NUM_T * NUM_ANT * NUM_K)

for n_sim_monto in range(NUM_MONTE_SIM):

    # for each monto carlo, generate a fixed channel realization
    g_mk = [[[np.array([0])] * NUM_ANT] * NUM_LAMBDA] * NUM_K  # pre-allocate a vector holding the channel realizations

    for _i_K, _k in enumerate(users):
        for _i_lmbda, lambda_corr in enumerate(lambdas):
            lambda_k = np.ones((_k, 1)) * lambda_corr
            lambda_compl_k = np.sqrt(1 - lambda_k ** 2)
            for _i_M, _m in enumerate(antennas):
                g_mk[_i_K][_i_lmbda][_i_M] = np.diag(lambda_compl_k[:, 0]) @ (
                        np.random.normal(0, 1 / np.sqrt(2), (_k, _m)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (_k, _m)))

    for n_sim_noise in range(NUM_NOISE_REALIZATIONS):
        for i_K, K in enumerate(users):
            ITER_MAX = K * 4
            rho = np.ones((K, 1)) * p_TX
            phi = np.random.uniform(0, 2 * np.pi, size=(K, 1))
            a = np.random.binomial(n=1, p=eps_a, size=(K, 1))
            if np.sum(a) < 1:
                a[0] = 1  # ensure_dir at least one user is active
            gamma = np.sqrt(rho) * a * np.exp(1j * phi)

            # x_int = np.random.randint(0, 4, (Td, K))  # 0 to 3
            # x_degrees = x_int * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
            # x_radians = x_degrees * np.pi / 180.0  # sin() and cos() takes in radians
            # payload = np.cos(x_radians) + 1j * np.sin(x_radians)  # this produces our QPSK complex symbols

            # payload = np.random.normal(0, 1 / np.sqrt(2), (Td, K)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (Td, K))

            for i_T, T in enumerate(preamble_lengths):

                ## Preamble generation and user activity
                s = np.random.normal(0, 1 / np.sqrt(2), (T, K)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (T, K))

                for i_lmbda, lambda_corr in enumerate(lambdas):
                    lambda_k = np.zeros((K, 1)) + lambda_corr
                    ## Channel generation
                    # lambda_k= beta_k - E(||g_k||^2/M)

                    for i_M, M in enumerate(antennas):

                        ## Channel generation (moved above to average over noise realizations)
                        # g = np.diag(lambda_compl_k[:, 0]) @ (
                        #         np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2),
                        #                                                                             (K, M)))
                        g = g_mk[i_K][i_lmbda][i_M]

                        for i_snr, snr in enumerate(snrs):

                            # gamma_real[n_sim, i_lmbda, i_snr, :] = gamma.copy().flatten()

                            epsilon = np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0,
                                                                                                          1 / np.sqrt(
                                                                                                              2),
                                                                                                          (K, M))
                            h = g + np.diag(lambda_k[:, 0]) @ epsilon

                            ## Received preamble
                            sigma2 = p_TX / snr
                            w = (np.random.normal(0, 1 / np.sqrt(2), (T, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2),
                                                                                                     (T, M))) * np.sqrt(
                                sigma2)
                            y = s @ np.diag(gamma[:, 0]) @ h + w

                            # w_payload = (np.random.normal(0, 1 / np.sqrt(2), (Td, M)) + 1j * np.random.normal(0,
                            #                                                                                   1 / np.sqrt(
                            #                                                                                       2),
                            #                                                                                   (Td,
                            #                                                                                    M))) * np.sqrt(
                            #     sigma2)
                            # y_payload = payload @ np.diag(gamma[:, 0]) @ h + w_payload
                            # Estimate based on prior csi, assuming all lambdas are zeros, cf. conf. paper

                            # input h or g? in prior CSI
                            gamma_hat_prior_CSI = utils.ZF(M, T, K, s, h, y)
                            gamma_hat_partial_CSI_ZF = utils.ZF(M, T, K, s, g, y)

                            # gamma_prior_csi[n_sim, i_lmbda, i_snr, i_T, :] = gamma_hat_prior_CSI.copy()[:, 0]

                            # C_inv_prior_CSI = utils.inv(
                            #     s @ np.diag(
                            #         np.abs(gamma_hat_prior_CSI[:, 0]) ** 2 * lambda_k[:,
                            #                                                  0] ** 2) @ s.T.conj() + sigma2 * np.identity(
                            #         T))
                            #
                            # C_inv_partial_CSI_ZF = utils.inv(
                            #     s @ np.diag(
                            #         np.abs(gamma_hat_prior_CSI[:, 0]) ** 2 * lambda_k[:,
                            #                                                  0] ** 2) @ s.T.conj() + sigma2 * np.identity(
                            #         T))

                            # MSE_prior_csi[i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma,
                            #                                                          gamma_hat_prior_CSI[:,
                            #                                                          0])  # utils.ML_value(gamma_prior_csi[n_sim, :], C_inv_prior_CSI, y, s, g, M, T)
                            # MSE_partial_csi_ZF[i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma,
                            #                                                               gamma_hat_partial_CSI_ZF[
                            #                                                               :,
                            #                                                               0])  # utils.ML_value(gamma_prior_csi[n_sim, :], C_inv_prior_CSI, y, s, g, M, T)

                            ## Estimator based on partial CSI and iterative ML
                            # Initialization thanks to prior CSI
                            gamma_hat_partial_CSI, C_inverse_partial_CSI, MSEs_partial = utils.algorithm(
                                gamma_hat_prior_CSI.copy(),
                                lambda_k, s,
                                M,
                                y, g,
                                sigma2,
                                T, K,
                                iter_max=ITER_MAX, real_gamma=gamma)
                            # gamma_partial_csi[n_sim, i_lmbda, i_snr, i_T, :] = gamma_hat_partial_CSI.copy()

                            # MSE_partial_csi[i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma,
                            #                                                            gamma_hat_partial_CSI)  # utils.ML_value(gamma_partial_csi[n_sim, :], C_inverse_partial_CSI, y, s, g, M, T)

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
                            gamma_hat_no_CSI, C_inverse_no_CSI, MSEs_no = utils.algorithm(np.zeros_like(gamma),
                                                                                          np.ones_like(lambda_k), s, M,
                                                                                          y,
                                                                                          np.zeros_like(g),
                                                                                          sigma2, T,
                                                                                          K, iter_max=ITER_MAX,
                                                                                          real_gamma=gamma)
                            # gamma_no_csi[n_sim, i_lmbda, i_snr, i_T:] = gamma_hat_no_CSI.copy()

                            # snr_k_no_CSI = (np.linalg.norm(g, axis=1) ** 2 + M * lambda_k ** 2) / sigma2

                            MSE_no_csi[i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma,
                                                                                  gamma_hat_no_CSI)  # utils.ML_value(gamma_no_csi[n_sim, :], C_inverse_no_CSI, y, s, g, M, T)

                            # fig = plt.figure()
                            # plt.plot(np.arange(len(MSEs_partial)) / K, 10 * np.log10(MSEs_partial), label="Partial CSI")
                            # plt.plot(np.arange(len(MSEs_partial)) / K, 10 * np.log10(MSEs_no), label="No CSI")
                            # plt.plot(np.arange(len(MSEs_partial)) / K, 10 * np.log10(
                            #     [MSE_prior_csi[n_sim, i_lmbda, i_snr, i_T, i_M, i_K]] * len(MSEs_partial)),
                            #          label="Prior CSI")
                            # plt.xlabel("# iterations / K")
                            # plt.ylabel("MSE")
                            # plt.legend()
                            # plt.tight_layout()
                            # plt.show()

                            # sum_rates = np.zeros(NUM_V, dtype=float)
                            # mses = np.zeros(NUM_V, dtype=float)
                            # correct_users = np.zeros(NUM_V, dtype=int)
                            # considered_active = np.zeros(NUM_V, dtype=int)
                            # phase_diff = np.abs(np.angle(gamma_hat_partial_CSI) - np.angle(gamma[:, 0]))
                            #
                            # fig = plt.figure()
                            # plt.title("Phase difference in degrees")
                            # plt.hist(phase_diff)
                            # plt.show()

                            for iv, v_dB in enumerate(np.linspace(-200, 20, num=NUM_V)):
                                v = 10 ** (v_dB / 10)

                                # force lowest v to be zero.
                                if iv == 0:
                                    v = 0.0

                                v_th = v / np.sqrt(snr)

                                the_slice = np.index_exp[i_lmbda, i_snr, i_T, i_M, i_K, iv]

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_prior_CSI) >= v_th] = 1

                                pa_prior_csi[the_slice] += utils.prob_false(a, act)
                                md_prior_csi[the_slice] += utils.prob_miss(a, act)

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_no_CSI) >= v_th] = 1

                                pa_no_csi[the_slice] += utils.prob_false(a, act)
                                md_no_csi[the_slice] += utils.prob_miss(a, act)

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_partial_CSI_ZF) >= v_th] = 1

                                pa_partial_csi_ZF[the_slice] += utils.prob_false(a, act)
                                md_partial_csi_ZF[the_slice] += utils.prob_miss(a, act)

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_partial_CSI) >= v_th] = 1

                                pa_partial_csi[the_slice] += utils.prob_false(a, act)
                                md_partial_csi[the_slice] += utils.prob_miss(a, act)

                                # detection
                                # Ka = act.copy()
                                # Ka = a.copy()
                                # # if no devices are considered active, than sum rate is zero
                                # if np.sum(Ka) == 0:
                                #     sum_rates[iv] = 0
                                #     continue
                                # # channel estimation
                                # # s€TxK,
                                # Ka_idx = np.flatnonzero(Ka)
                                # p = s[:, Ka_idx].T  # preambles from considered active users
                                # p_H = p.T.conj()
                                # # transpose y to have right dimensions
                                # h_est = y.T @ p_H @ np.linalg.inv(p @ p_H)
                                # # not the MSE operates on flattened array, thus average MSE over all users
                                # mses[iv] = utils.MSE_dB(h[Ka_idx, :].T, h_est)
                                # # decoding
                                # h_a = h[Ka_idx, :].T

                                # G = g.T[:, Ka_idx].copy()
                                # S_a = s.T[Ka_idx, :].copy()
                                # Y = y.T.copy()
                                # gamma_diag = np.diag(gamma_hat_partial_CSI[Ka_idx]).copy()
                                # S_a_H = S_a.conj().T.copy()
                                # S_a_inverse = S_a_H @ np.linalg.inv(S_a @ S_a_H)
                                # E = (Y - G @ S_a) @ S_a_inverse
                                # H = G + E
                                #
                                # mses[iv] = utils.MSE_dB(h[Ka_idx, :].T, H)

                                # w = np.linalg.inv(h_a.conj().T @ h_a) @ h_a.conj().T
                                # rx_payload = w @ y_payload.T
                                # # snr
                                # mse_payload = utils.MSE_dB(payload[:, Ka_idx].T, rx_payload)
                                # sinr_payload = utils.SINR_dB(payload[:, Ka_idx].T, rx_payload)
                                # # compute sum rate based on correct detected users
                                # # looking for the Ka_idx which are correct
                                # # correct idx are at places where a*Ka = 1
                                # Ka_union = a.flatten() * Ka.flatten()
                                # Ka_union_idx = np.flatnonzero(Ka_union)
                                # Ka_correct, Ka_idx_correct, _ = np.intersect1d(Ka_idx, Ka_union_idx, return_indices=True)
                                # # SINR of the found active users (excluding perceived active but non-active users)
                                # sum_rates[iv] = utils.sum_rate(payload[:, Ka_union_idx].T, rx_payload[Ka_idx_correct, :])
                                # correct_users[iv] = len(Ka_correct)
                                # considered_active[iv] = np.sum(Ka)
                            # plt.plot(sum_rates, label="R")
                            # fig = plt.figure()
                            # plt.plot(mses, label="MSE")
                            # plt.plot(correct_users, label="# correct active users")
                            # plt.plot(considered_active, label="# considered active users")
                            # plt.legend()
                            # plt.show()
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

# average over all simulations (monto + noise)
pa_prior_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_prior_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

pa_partial_csi_ZF /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_partial_csi_ZF /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

pa_partial_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_partial_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

pa_no_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_no_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

np.savez(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"), pa_prior_csi=pa_prior_csi,
         md_prior_csi=md_prior_csi, pa_partial_csi_ZF=pa_partial_csi_ZF, md_partial_csi_ZF=md_partial_csi_ZF,
         pa_partial_csi=pa_partial_csi, md_partial_csi=md_partial_csi,
         pa_no_csi=pa_no_csi, md_no_csi=md_no_csi)

mean_axis = tuple(range(pa_prior_csi.ndim - 1))
fig = plt.figure()
plt.plot(pa_prior_csi.mean(axis=mean_axis), md_prior_csi.mean(axis=mean_axis), label="Full CSI (ZF)", marker="x")
plt.plot(pa_partial_csi_ZF.mean(axis=mean_axis), md_partial_csi_ZF.mean(axis=mean_axis), label="Partial CSI (ZF)",
         marker="x")
plt.plot(pa_partial_csi.mean(axis=mean_axis), md_partial_csi.mean(axis=mean_axis), label="Partial CSI (algo)",
         marker="x")
plt.plot(pa_no_csi.mean(axis=mean_axis), md_no_csi.mean(axis=mean_axis), label="No CSI (algo)", marker="x")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("FA")
plt.ylabel("MD")
plt.legend()
plt.tight_layout()
plt.show()

pbar.close()

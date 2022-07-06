import os
import warnings

import keyboard
import matplotlib

matplotlib.use('Qt5Agg')

import utils
import tqdm
import matplotlib.pyplot as plt

import numpy as np
from numba.core.errors import NumbaPerformanceWarning

# import jax.numpy as np

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# plt.close('all')

# K = 40  # Number of single-antenna users
# M = 64  # Number of receive antennas


# beta_k = np.ones((K, 1))
eps_a = 0.1

NUM_NOISE_REALIZATIONS = 50
NUM_LAMBDA = 2
NUM_SNR = 20
NUM_T = 1  # number of diff preambles per run 10->40

NUM_V = 1000

lambdas = np.linspace(0.3, 0.9, num=NUM_LAMBDA)
preamble_lengths = np.linspace(10, 40, num=NUM_T).astype(int)

snrs_dB = np.linspace(20, -20, num=NUM_SNR)
snrs = 10 ** (np.asarray(snrs_dB) / 10)
antennas = [64]

p_TX = 1

users = [500]  # Number of single-antenna users

params = {
    "lambdas": lambdas,
    "preamble_lengths": preamble_lengths,
    "snrs_dB": snrs_dB,
    "antennas": antennas,
    "users": users
}

NUM_ANT = len(antennas)
NUM_K = len(users)

SHAPE_PROB = (NUM_LAMBDA, NUM_SNR, NUM_T, NUM_ANT, NUM_K, NUM_V)

pa_prior_csi = np.zeros(SHAPE_PROB, dtype=float)
md_prior_csi = np.zeros(SHAPE_PROB, dtype=float)

# pa_no_csi = np.zeros(SHAPE_PROB, dtype=float)
# md_no_csi = np.zeros(SHAPE_PROB, dtype=float)

pa_partial_csi = np.zeros(SHAPE_PROB, dtype=float)
md_partial_csi = np.zeros(SHAPE_PROB, dtype=float)

pa_partial_csi_RZF = np.zeros(SHAPE_PROB, dtype=float)
md_partial_csi_RZF = np.zeros(SHAPE_PROB, dtype=float)

n_sim_monto = 0

while True:

    if keyboard.is_pressed("q"):
        break

    print(f"Starting simulation {n_sim_monto}", flush=True)
    pbar = tqdm.tqdm(total=NUM_NOISE_REALIZATIONS * NUM_LAMBDA * NUM_SNR * NUM_T * NUM_ANT * NUM_K)
    # for each monto carlo, generate a fixed channel realization
    g_mk = {}  # pre-allocate a vector holding the channel realizations

    for _i_K, _k in enumerate(users):
        g_mk[_i_K] = {}
        for _i_lmbda, lambda_corr in enumerate(lambdas):
            g_mk[_i_K][_i_lmbda] = {}
            lambda_k = np.ones((_k, 1)) * lambda_corr
            lambda_compl_k = np.sqrt(1 - lambda_k ** 2)
            for _i_M, _m in enumerate(antennas):
                g_mk[_i_K][_i_lmbda][_i_M] = np.diag(lambda_compl_k[:, 0]) @ (
                        np.random.normal(0, 1 / np.sqrt(2), (_k, _m)) + 1j * np.random.normal(0, 1 / np.sqrt(2),
                                                                                              (_k, _m)))

    for n_sim_noise in range(NUM_NOISE_REALIZATIONS):
        for i_K, K in enumerate(users):
            ITER_MAX = K * 4
            rho = np.ones((K, 1)) * p_TX
            phi = np.random.uniform(0, 2 * np.pi, size=(K, 1))
            a = np.random.binomial(n=1, p=eps_a, size=(K, 1))
            if np.sum(a) < 1:
                a[0] = 1  # ensure_dir at least one user is active
            gamma = np.sqrt(rho) * a * np.exp(1j * phi)

            for i_T, T in enumerate(preamble_lengths):

                ## Preamble generation and user activity
                s = np.random.normal(0, 1 / np.sqrt(2), (T, K)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (T, K))

                for i_lmbda, lambda_corr in enumerate(lambdas):
                    lambda_k = np.zeros((K, 1)) + lambda_corr

                    for i_M, M in enumerate(antennas):
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

                            gamma_hat_prior_CSI_RZF = utils.RZF(M, T, K, s, h, y, sigma2, eps_a, p_TX)
                            gamma_hat_partial_CSI_RZF = utils.RZF(M, T, K, s, g, y, sigma2, eps_a, p_TX)

                            ## Estimator based on partial CSI and iterative ML

                            gamma_init = gamma_hat_partial_CSI_RZF.copy()
                            # Initialization thanks to prior CSI
                            gamma_hat_partial_CSI, C_inverse_partial_CSI, (MSEs_partial, LLs_partial) = utils.algorithm(
                                gamma_init,
                                lambda_k, s,
                                M,
                                y, g,
                                sigma2,
                                T, K,
                                iter_max=ITER_MAX, real_gamma=gamma)

                            # import matplotlib.pyplot as plt
                            # fig, ax_ll = plt.subplots()
                            # ax_mse = ax_ll.twinx()
                            # ax3 = ax_mse.twiny()
                            # ax_ll.get_shared_x_axes().join(ax_mse, ax3)
                            #
                            # ax_ll.set_ylabel("LLs", color="blue")
                            # ax_mse.set_ylabel("MSEs", color="red")
                            #
                            # ax_ll.plot(LLs_partial, label="LLs", color="blue")
                            # ax_mse.plot(MSEs_partial, label="MSEs", color="red")
                            # plt.tight_layout()
                            # plt.show()

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
                            # gamma_hat_no_CSI, C_inverse_no_CSI, MSEs_no = utils.algorithm_no_csi(
                            #     np.ones_like(gamma_init), s, M,
                            #     y,
                            #     sigma2, T,
                            #     K, iter_max=ITER_MAX,
                            #     real_gamma=gamma)
                            # gamma_no_csi[n_sim, i_lmbda, i_snr, i_T:] = gamma_hat_no_CSI.copy()

                            # snr_k_no_CSI = (np.linalg.norm(g, axis=1) ** 2 + M * lambda_k ** 2) / sigma2

                            # MSE_no_csi[i_lmbda, i_snr, i_T, i_M, i_K] = utils.MSE(gamma,
                            #                                                       gamma_hat_no_CSI)  # utils.ML_value(gamma_no_csi[n_sim, :], C_inverse_no_CSI, y, s, g, M, T)

                            # MSE_rzf = utils.MSE(gamma, gamma_hat_partial_CSI_RZF)
                            # fig = plt.figure()
                            # plt.plot(np.arange(len(MSEs_partial)) / K, MSEs_partial, label="Partial CSI")
                            # plt.scatter(ITER_MAX / K, MSE_rzf, label="RZF Partial CSI")
                            # # plt.plot(np.arange(len(MSEs_partial)) / K, 10 * np.log10(MSEs_no), label="No CSI")
                            # # plt.plot(np.arange(len(MSEs_partial)) / K, 10 * np.log10(
                            # #     [MSE_prior_csi[n_sim, i_lmbda, i_snr, i_T, i_M, i_K]] * len(MSEs_partial)),
                            # #          label="Prior CSI")
                            # plt.xlabel("# iterations / K")
                            # plt.ylabel("MSE value")
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
                            #
                            # fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
                            # axs[0].stem(np.abs(gamma_hat_prior_CSI), label="prior CSI")
                            # axs[1].stem(np.abs(gamma_hat_partial_CSI), label="partial CSI")
                            # axs[2].stem(np.abs(gamma_hat_partial_CSI_ZF), label="prior CSI")
                            # axs[3].stem(np.abs(gamma_hat_no_CSI), label="no CSI")
                            # axs[4].stem(np.abs(gamma[:, 0]), label="real")
                            # plt.legend()
                            # plt.show()

                            v_snr = np.array(
                                [np.linalg.norm(g[k, :]) ** 2 + M * lambda_k[k, 0] ** 2 for k in
                                 range(K)]).flatten() / sigma2
                            for iv, v_dB in enumerate(np.linspace(-40, 40, num=NUM_V)):
                                v = 10 ** (v_dB / 10)

                                # force lowest v to be zero.
                                if iv == 0:
                                    v = -1.0

                                v_th = v / np.sqrt(v_snr)

                                the_slice = np.index_exp[i_lmbda, i_snr, i_T, i_M, i_K, iv]

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_prior_CSI_RZF.flatten()) >= v_th] = 1

                                pa_prior_csi[the_slice] += utils.prob_false(a, act)
                                md_prior_csi[the_slice] += utils.prob_miss(a, act)

                                # act = np.zeros_like(a)
                                # act[np.abs(gamma_hat_no_CSI) >= v_th] = 1
                                #
                                # pa_no_csi[the_slice] += utils.prob_false(a, act)
                                # md_no_csi[the_slice] += utils.prob_miss(a, act)

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_partial_CSI_RZF.flatten()) >= v_th] = 1

                                pa_partial_csi_RZF[the_slice] += utils.prob_false(a, act)
                                md_partial_csi_RZF[the_slice] += utils.prob_miss(a, act)

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_partial_CSI.flatten()) >= v_th] = 1

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

    n_sim_monto += 1
    pbar.close()

# average over all simulations (monto + noise)
pa_prior_csi /= (n_sim_monto * NUM_NOISE_REALIZATIONS)
md_prior_csi /= (n_sim_monto * NUM_NOISE_REALIZATIONS)

pa_partial_csi_RZF /= (n_sim_monto * NUM_NOISE_REALIZATIONS)
md_partial_csi_RZF /= (n_sim_monto * NUM_NOISE_REALIZATIONS)

pa_partial_csi /= (n_sim_monto * NUM_NOISE_REALIZATIONS)
md_partial_csi /= (n_sim_monto * NUM_NOISE_REALIZATIONS)

# pa_no_csi /= (n_sim_monto * NUM_NOISE_REALIZATIONS)
# md_no_csi /= (n_sim_monto * NUM_NOISE_REALIZATIONS)

import secrets

results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

print(f"Pressed 'q', stopping... #simulations is {n_sim_monto}")
outfile = f"data-{secrets.token_urlsafe(16)}"

print(f"Saving to {outfile}")
np.savez_compressed(os.path.join(results_dir, outfile),
                    pa_prior_csi=pa_prior_csi,
                    md_prior_csi=md_prior_csi, pa_partial_csi_ZF=pa_partial_csi_RZF,
                    md_partial_csi_ZF=md_partial_csi_RZF,
                    pa_partial_csi=pa_partial_csi, md_partial_csi=md_partial_csi,
                    params=params, SHAPE_PROB=SHAPE_PROB)

import os
import warnings

# import matplotlib
#
# matplotlib.use('Qt5Agg')

import utils
import tqdm
# import matplotlib.pyplot as plt

import numpy as np
from numba.core.errors import NumbaPerformanceWarning

# import jax.numpy as np

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

np.random.seed(666)

p_TX = 1

eps_a = 0.1

NUM_MONTE_SIM = 10
NUM_NOISE_REALIZATIONS = 10000
NUM_LAMBDA = 10
NUM_SNR = 2
NUM_T = 1  # number of diff preambles per run 10->40

NUM_V = 200

lambdas = np.linspace(0.99, 0.1, num=NUM_LAMBDA)
preamble_lengths = np.linspace(10, 40, num=NUM_T).astype(int)

snrs_dB = np.linspace(-20, 20, num=NUM_SNR)
snrs = 10 ** (np.asarray(snrs_dB) / 10)
antennas = [128]  # [32, 64, 128]

users = [100]  # [60, 100, 120, 200, 500]  # Number of single-antenna users

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

pa_no_csi = np.zeros(SHAPE_PROB, dtype=float)
md_no_csi = np.zeros(SHAPE_PROB, dtype=float)

pa_partial_csi = np.zeros(SHAPE_PROB, dtype=float)
md_partial_csi = np.zeros(SHAPE_PROB, dtype=float)

pa_partial_csi_ZF = np.zeros(SHAPE_PROB, dtype=float)
md_partial_csi_ZF = np.zeros(SHAPE_PROB, dtype=float)


pbar = tqdm.tqdm(total=NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS * NUM_LAMBDA * NUM_SNR * NUM_T * NUM_ANT * NUM_K)

for n_sim_monto in range(NUM_MONTE_SIM):

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
                    ## Channel generation
                    # lambda_k= beta_k - E(||g_k||^2/M)

                    for i_M, M in enumerate(antennas):

                        g = g_mk[i_K][i_lmbda][i_M]

                        for i_snr, snr in enumerate(snrs):

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

                            # gamma_hat_prior_CSI = utils.ZF(M, T, K, s, h, y)
                            # gamma_hat_partial_CSI_ZF = utils.ZF(M, T, K, s, g, y)

                            #gamma_hat_partial_CSI_MF = utils.MF(M, T, K, s, g, y, eps_a, p_TX, sigma2)

                            ## Estimator based on partial CSI and iterative ML

                            gamma_init = np.zeros_like(gamma)  # np.zeros_like(gamma)  # or gamma_hat_partial_CSI_ZF.copy()
                            # Initialization thanks to prior CSI
                            gamma_hat_partial_CSI, C_inverse_partial_CSI, MSEs_partial = utils.algorithm(
                                gamma_init,
                                lambda_k, s,
                                M,
                                y, g,
                                sigma2,
                                T, K,
                                iter_max=ITER_MAX, real_gamma=gamma)

                            # Estimator based on no CSI and iterative ML (as Caire)
                            gamma_hat_no_CSI, C_inverse_no_CSI, MSEs_no = utils.algorithm_no_csi(
                                np.zeros_like(gamma_init), s, M,
                                y,
                                sigma2, T,
                                K, iter_max=ITER_MAX,
                                real_gamma=gamma)

                            for iv, v_dB in enumerate(np.linspace(-40, 40, num=NUM_V)):
                                v = 10 ** (v_dB / 10)

                                # force lowest v to be zero.
                                if iv == 0:
                                    v = 0.0

                                v_th = v / np.sqrt(snr)

                                the_slice = np.index_exp[i_lmbda, i_snr, i_T, i_M, i_K, iv]

                                # act = np.zeros_like(a)
                                # act[np.abs(gamma_hat_prior_CSI) >= v_th] = 1
                                #
                                # pa_prior_csi[the_slice] += utils.prob_false(a, act)
                                # md_prior_csi[the_slice] += utils.prob_miss(a, act)

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_no_CSI) >= v_th] = 1

                                pa_no_csi[the_slice] += utils.prob_false(a, act)
                                md_no_csi[the_slice] += utils.prob_miss(a, act)

                                # act = np.zeros_like(a)
                                # act[np.abs(gamma_hat_partial_CSI_ZF) >= v_th] = 1
                                #
                                # pa_partial_csi_ZF[the_slice] += utils.prob_false(a, act)
                                # md_partial_csi_ZF[the_slice] += utils.prob_miss(a, act)

                                act = np.zeros_like(a)
                                act[np.abs(gamma_hat_partial_CSI) >= v_th] = 1

                                pa_partial_csi[the_slice] += utils.prob_false(a, act)
                                md_partial_csi[the_slice] += utils.prob_miss(a, act)

                            pbar.update()

# average over all simulations (monto + noise)
pa_prior_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_prior_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

pa_partial_csi_ZF /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_partial_csi_ZF /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

pa_partial_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_partial_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

pa_no_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)
md_no_csi /= (NUM_MONTE_SIM * NUM_NOISE_REALIZATIONS)

np.savez_compressed(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"), pa_prior_csi=pa_prior_csi,
                    md_prior_csi=md_prior_csi, pa_partial_csi_ZF=pa_partial_csi_ZF, md_partial_csi_ZF=md_partial_csi_ZF,
                    pa_partial_csi=pa_partial_csi, md_partial_csi=md_partial_csi,
                    pa_no_csi=pa_no_csi, md_no_csi=md_no_csi, params=params, SHAPE_PROB=SHAPE_PROB)

pbar.close()

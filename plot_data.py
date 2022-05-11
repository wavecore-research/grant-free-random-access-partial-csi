import os
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np


def same_prob_args(x1, x2):
    """
    Find the same probability of MD and FA
    This makes it easier to plot wrt SNR/preamble length/...
    """
    assert x1.shape == x2.shape, "Both matrices should have the same shape"

    argmin = np.argmin(np.abs(np.abs(x1) - np.abs(x2)), axis=-1)[:, :, :, :, :,
             np.newaxis]  # new axis is to keep the dimensions/shape of x1
    # when numba supports numpy>1.22 we could change by using keepdims=True
    return np.take_along_axis(x1, argmin, axis=-1), np.take_along_axis(x2, argmin, axis=-1), argmin


with np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data.npz"), allow_pickle=True) as data:
    pa_prior_csi = data["pa_prior_csi"]
    md_prior_csi = data["md_prior_csi"]
    pa_partial_csi_ZF = data["pa_partial_csi_ZF"]
    md_partial_csi_ZF = data["md_partial_csi_ZF"]
    pa_partial_csi = data["pa_partial_csi"]
    md_partial_csi = data["md_partial_csi"]
    pa_no_csi = data["pa_no_csi"]
    md_no_csi = data["md_no_csi"]
    params = data["params"].item()
    SHAPE_PROB = data["SHAPE_PROB"]

    # SHAPE = (NUM_LAMBDA, NUM_SNR, NUM_T, NUM_ANT, NUM_K, NUM_V)

    # params = {
    #     "lambdas": lambdas,
    #     "preamble_lengths":preamble_lengths,
    #     "snrs_dB":snrs_dB,
    #     "antennas":antennas,
    #     "users":users
    # }

    snrs_db = params["snrs_dB"]
    snrs = 10 ** (np.asarray(snrs_db) / 10)

    mean_axis = tuple(range(pa_prior_csi.ndim - 1))
    plt.figure()
    # plt.plot(pa_prior_csi.mean(axis=mean_axis), md_prior_csi.mean(axis=mean_axis), label="Full CSI (ZF)", marker="x")
    x1, x2, args = same_prob_args(pa_prior_csi, md_prior_csi)

    plt.plot(snrs_db, x1[0, :, 0, 0, 0], label="Full CSI (ZF)")

    x1, x2, args = same_prob_args(pa_partial_csi_ZF, md_partial_csi_ZF)
    plt.plot(snrs_db, x1[0, :, 0, 0, 0], label="Partial CSI (ZF)")

    x1, x2, args = same_prob_args(pa_partial_csi, md_partial_csi)
    plt.plot(snrs_db, x1[0, :, 0, 0, 0], label="Partial CSI (algo)")

    # x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    # plt.scatter(snrs_db, x1[0, :, 0, 0, 0], label="No CSI (algo) x1")
    # plt.scatter(snrs_db, x2[0, :, 0, 0, 0], label="No CSI (algo) x2")


    plt.yscale("log")
    plt.xlabel("SNR")
    plt.ylabel("Prob")
    plt.legend()
    plt.tight_layout()
    plt.show()

    Ms = params["antennas"]
    plt.figure()
    # plt.plot(pa_prior_csi.mean(axis=mean_axis), md_prior_csi.mean(axis=mean_axis), label="Full CSI (ZF)", marker="x")
    x1, x2, args = same_prob_args(pa_prior_csi, md_prior_csi)

    plt.plot(Ms, x2[0, 0, 0, :, 0], label="Full CSI (ZF)")

    x1, x2, args = same_prob_args(pa_partial_csi_ZF, md_partial_csi_ZF)
    plt.plot(Ms, x2[0, 0, 0, :, 0], label="Partial CSI (ZF)")

    x1, x2, args = same_prob_args(pa_partial_csi, md_partial_csi)
    plt.plot(Ms, x2[0, 0, 0, :, 0], label="Partial CSI (algo)")

    x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    plt.plot(Ms, x2[0, 0, 0, :, 0], label="No CSI (algo)")

    # x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    # plt.scatter(snrs_db, x1[0, :, 0, 0, 0], label="No CSI (algo) x1")
    # plt.scatter(snrs_db, x2[0, :, 0, 0, 0], label="No CSI (algo) x2")

    plt.yscale("log")
    plt.xlabel("M")
    plt.ylabel("Prob")
    plt.legend()
    plt.tight_layout()
    plt.show()

    lambdas = params["lambdas"]
    plt.figure()
    # plt.plot(pa_prior_csi.mean(axis=mean_axis), md_prior_csi.mean(axis=mean_axis), label="Full CSI (ZF)", marker="x")
    x1, x2, args = same_prob_args(pa_prior_csi, md_prior_csi)

    plt.plot(lambdas, x1[:, 0, 0, 0, 0], label="Full CSI (ZF)")

    x1, x2, args = same_prob_args(pa_partial_csi_ZF, md_partial_csi_ZF)
    plt.plot(lambdas, x1[:, 0, 0, 0, 0], label="Partial CSI (ZF)")

    x1, x2, args = same_prob_args(pa_partial_csi, md_partial_csi)
    plt.plot(lambdas, x1[:, 0, 0, 0, 0], label="Partial CSI (algo)")

    x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    plt.plot(lambdas, x1[:, 0, 0, 0, 0], label="No CSI (algo)")

    # x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    # plt.scatter(snrs_db, x1[0, :, 0, 0, 0], label="No CSI (algo) x1")
    # plt.scatter(snrs_db, x2[0, :, 0, 0, 0], label="No CSI (algo) x2")

    plt.yscale("log")
    plt.xlabel("$\lambda$")
    plt.ylabel("Prob")
    plt.legend()
    plt.tight_layout()
    plt.show()

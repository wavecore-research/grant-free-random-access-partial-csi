import os
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np

plt.close("all")


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


with np.load(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", "data-DD4eslEKZEFX_SNCadPEVg.npz"),
        allow_pickle=True) as data:
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
    lambdas = params["lambdas"]
    Ts = params["preamble_lengths"]
    Ms = params["antennas"]
    Ks = params["users"]

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

    LAMBDA_IDX = 0
    SNR_IDX = 1
    T_IDX = 2
    M_IDX = 3
    K_IDX = 4

    LOWEST_SNR_IDX = np.argmin(snrs_db)
    HIGHEST_SNR_IDX = np.argmax(snrs_db)

    LOWEST_M_IDX = np.argmin(params["antennas"])
    HIGHEST_M_IDX = np.argmax(params["antennas"])

    HIGHEST_K_IDX = np.argmax(params["users"])

    LOWEST_T_IDX = np.argmin(params["preamble_lengths"])
    HIGHEST_T_IDX = np.argmax(params["preamble_lengths"])

    HIGHEST_LAMBDA_IDX = np.argmax(params["lambdas"])
    LOWEST_LAMBDA_IDX = np.argmin(params["lambdas"])

    fig = plt.figure()
    selection = np.index_exp[LOWEST_LAMBDA_IDX, LOWEST_SNR_IDX, LOWEST_T_IDX, LOWEST_M_IDX, HIGHEST_K_IDX, :]

    plt.plot(pa_prior_csi[selection], md_prior_csi[selection], label="Full CSI (RZF)", marker="x")
    plt.plot(pa_partial_csi_ZF[selection], md_partial_csi_ZF[selection], label="Partial CSI (RZF)",
             marker="x")
    plt.plot(pa_partial_csi[selection], md_partial_csi[selection], label="Partial CSI (algo)",
             marker="x")
    # plt.plot(pa_no_csi[selection], md_no_csi[selection], label="No CSI (algo)", marker="x")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FA")
    plt.ylabel("MD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    # plt.plot(pa_prior_csi.mean(axis=mean_axis), md_prior_csi.mean(axis=mean_axis), label="Full CSI (ZF)", marker="x")
    x1, x2, args = same_prob_args(pa_prior_csi, md_prior_csi)

    selection = np.index_exp[HIGHEST_LAMBDA_IDX, :, HIGHEST_T_IDX, HIGHEST_M_IDX, HIGHEST_K_IDX]

    plt.plot(snrs_db, x1[selection], label="Full CSI (RZF)")

    x1, x2, args = same_prob_args(pa_partial_csi_ZF, md_partial_csi_ZF)
    plt.plot(snrs_db, x1[selection], label=f"Partial CSI (RZF) {lambdas[HIGHEST_LAMBDA_IDX]:.2f}")

    x1, x2, args = same_prob_args(pa_partial_csi, md_partial_csi)
    plt.plot(snrs_db, x1[selection], label=f"Partial CSI (algo) {lambdas[HIGHEST_LAMBDA_IDX]:.2f}")

    selection = np.index_exp[LOWEST_LAMBDA_IDX, :, LOWEST_T_IDX, HIGHEST_M_IDX, HIGHEST_K_IDX]
    x1, x2, args = same_prob_args(pa_partial_csi_ZF, md_partial_csi_ZF)
    plt.plot(snrs_db, x1[selection], label=f"Partial CSI (RZF) {lambdas[LOWEST_LAMBDA_IDX]:.2f}")

    x1, x2, args = same_prob_args(pa_partial_csi, md_partial_csi)
    plt.plot(snrs_db, x1[selection], label=f"Partial CSI (algo) {lambdas[LOWEST_LAMBDA_IDX]:.2f}")

    # x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    # plt.scatter(snrs_db, x1[0, :, 0, 0, 0], label="No CSI (algo) x1")
    # plt.scatter(snrs_db, x2[0, :, 0, 0, 0], label="No CSI (algo) x2")

    plt.yscale("log")
    plt.xlabel("SNR")
    plt.ylabel("Prob")
    plt.title(f"Params $T$={Ts[selection[T_IDX]]}, $M$={Ms[selection[M_IDX]]}, $K$={Ks[selection[K_IDX]]}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    # plt.plot(pa_prior_csi.mean(axis=mean_axis), md_prior_csi.mean(axis=mean_axis), label="Full CSI (ZF)", marker="x")
    x1, x2, args = same_prob_args(pa_prior_csi, md_prior_csi)

    selection = np.index_exp[HIGHEST_LAMBDA_IDX, LOWEST_SNR_IDX, LOWEST_T_IDX, :, HIGHEST_K_IDX]

    plt.plot(Ms, x2[selection], label="Full CSI (RZF)")

    x1, x2, args = same_prob_args(pa_partial_csi_ZF, md_partial_csi_ZF)
    plt.plot(Ms, x2[selection], label="Partial CSI (RZF)")

    x1, x2, args = same_prob_args(pa_partial_csi, md_partial_csi)
    plt.plot(Ms, x2[selection], label="Partial CSI (algo)")

    x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    plt.plot(Ms, x2[selection], label="No CSI (algo)")

    # x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    # plt.scatter(snrs_db, x1[0, :, 0, 0, 0], label="No CSI (algo) x1")
    # plt.scatter(snrs_db, x2[0, :, 0, 0, 0], label="No CSI (algo) x2")

    plt.yscale("log")
    plt.xlabel("M")
    plt.ylabel("Prob")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (NUM_LAMBDA, NUM_SNR, NUM_T, NUM_ANT, NUM_K, NUM_V)

    plt.figure()
    # plt.plot(pa_prior_csi.mean(axis=mean_axis), md_prior_csi.mean(axis=mean_axis), label="Full CSI (ZF)", marker="x")

    selection = np.index_exp[:, HIGHEST_SNR_IDX, LOWEST_T_IDX, LOWEST_M_IDX, HIGHEST_K_IDX]

    x1, x2, args = same_prob_args(pa_prior_csi, md_prior_csi)
    plt.plot(lambdas, x1[selection], label="Full CSI (RZF)")

    x1, x2, args = same_prob_args(pa_partial_csi_ZF, md_partial_csi_ZF)
    plt.plot(lambdas, x1[selection], label="Partial CSI (RZF)")

    x1, x2, args = same_prob_args(pa_partial_csi, md_partial_csi)
    plt.plot(lambdas, x1[selection], label="Partial CSI (algo)")

    x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    plt.plot(lambdas, x1[selection], label="No CSI (algo)")

    # x1, x2, args = same_prob_args(pa_no_csi, md_no_csi)
    # plt.scatter(snrs_db, x1[0, :, 0, 0, 0], label="No CSI (algo) x1")
    # plt.scatter(snrs_db, x2[0, :, 0, 0, 0], label="No CSI (algo) x2")

    plt.yscale("log")
    plt.xlabel("$\lambda$")
    plt.ylabel("Prob")
    plt.legend()
    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap('viridis')

    colors = [cmap(x) for x in np.linspace(0, 0.8, num=4)]

    linestyles = [
        'solid',  # Same as (0, ()) or '-'
        'dotted',  # Same as (0, (1, 1)) or ':'
        'dashed',  # Same as '--'
        'dashdot']
    fig, ax = plt.subplots()

    i_lambdas = range(len(lambdas))
    #i_lambdas = [1, 4, 7]
    for iil, il in enumerate(i_lambdas):
        selection = np.index_exp[il, HIGHEST_SNR_IDX, LOWEST_T_IDX, HIGHEST_M_IDX, HIGHEST_K_IDX]
        ax.plot(pa_prior_csi[selection], md_prior_csi[selection], label=lambdas[il], color=f"C{iil}", ls=linestyles[0])
        ax.plot(pa_partial_csi_ZF[selection], md_partial_csi_ZF[selection], color=f"C{iil}", ls=linestyles[1])
        ax.plot(pa_partial_csi[selection], md_partial_csi[selection], color=f"C{iil}", ls=linestyles[2])
        # ax.plot(pa_no_csi[selection], md_no_csi[selection], color=f"C{iil}", ls=linestyles[3])

    lines = ax.get_lines()
    legend1 = plt.legend([lines[i] for i in [0, 1, 2]],
                         ["Full CSI (RZF)", "Partial CSI (RZF)", "Partial CSI (algo)"], loc=3, )
    legend2 = plt.legend([lines[i] for i in [j * 3 for j in range(len(i_lambdas))]],
                         [f"{lambdas[il]:.2f} ({(1 - lambdas[il] ** 2) * 100:.1f}%)" for il in i_lambdas], loc=8,
                         title="$\lambda$ (1-$\lambda^2$%)")
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FA")
    plt.ylabel("MD")
    plt.tight_layout()
    plt.show()

    colors = [cmap(x) for x in np.linspace(0, 0.8, num=4)]

    linestyles = [
        'solid',  # Same as (0, ()) or '-'
        'dotted',  # Same as (0, (1, 1)) or ':'
        'dashed',  # Same as '--'
        'dashdot']
    fig, ax = plt.subplots()

    i_snrs = range(len(snrs))
    #i_snrs = [1, 4, 7]
    for iil, il in enumerate(i_snrs):
        selection = np.index_exp[LOWEST_LAMBDA_IDX, il, LOWEST_T_IDX, HIGHEST_M_IDX, HIGHEST_K_IDX]
        ax.plot(pa_prior_csi[selection], md_prior_csi[selection], label=f"{snrs_db[il]:.2f}dB", color=f"C{iil}", ls=linestyles[0])
        ax.plot(pa_partial_csi_ZF[selection], md_partial_csi_ZF[selection], color=f"C{iil}", ls=linestyles[1])
        ax.plot(pa_partial_csi[selection], md_partial_csi[selection], color=f"C{iil}", ls=linestyles[2])
        # ax.plot(pa_no_csi[selection], md_no_csi[selection], color=f"C{iil}", ls=linestyles[3])

    lines = ax.get_lines()
    legend1 = plt.legend([lines[i] for i in [0, 1, 2]],
                         ["Full CSI (RZF)", "Partial CSI (RZF)", "Partial CSI (algo)"], loc=3, )
    legend2 = plt.legend([lines[i] for i in [j * 3 for j in range(len(i_snrs))]],
                         [f"{snrs_db[il]:.2f}dB" for il in i_snrs], loc=8,
                         title="SNR")
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FA")
    plt.ylabel("MD")
    plt.tight_layout()
    plt.show()

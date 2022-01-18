import matplotlib.pyplot as plt
import numpy as np

import utils

plt.close('all')

K = 40  # Number of single-antenna users
M = 64  # Number of receive antennas
T = 10  # Preamble length
p_TX = 1
SNR_dB = 20
SNR = 10 ** (SNR_dB / 10)
beta_k = np.ones((K, 1))

ITER_MAX = K * 10
np.random.seed(0)

## Preamble generation and user activity
s = np.random.normal(0, 1 / np.sqrt(2), (T, K)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (T, K))
a = np.random.binomial(n=1, p=0.1, size=(K, 1))
phi = np.random.uniform(0, 2 * np.pi, size=(K, 1))
rho = np.ones((K, 1)) * p_TX
gamma = np.sqrt(rho) * a * np.exp(1j * phi)

## Channel generation
# lambda_k=np.random.uniform(0,1,(K,1))
# TODO lambda_k= beta_k - E(||g_k||^2/M)
lambda_k = np.zeros((K, 1)) + 0.95
lambda_compl_k = np.sqrt(1 - lambda_k ** 2)
# lambda_compl_k=np.ones((K,1))*0.7

g = np.diag(lambda_compl_k[:, 0]) @ (
        np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (K, M)))
epsilon = np.random.normal(0, 1 / np.sqrt(2), (K, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (K, M))

h = g + np.diag(lambda_k[:, 0]) @ epsilon

## Received preamble
sigma2 = p_TX / SNR
w = (np.random.normal(0, 1 / np.sqrt(2), (T, M)) + 1j * np.random.normal(0, 1 / np.sqrt(2), (T, M))) * np.sqrt(sigma2)
y = s @ np.diag(gamma[:, 0]) @ h + w

# Estimate based on prior csi, assuming all lambdas are zeros, cf. conf. paper
y_tilde = np.reshape(y.T, (M * T, 1))
D = np.diag(s.reshape(K * T)) @ np.kron(np.ones((T, 1)), np.identity(K))
Gamma = np.zeros((M * T, K), dtype=complex)
for index_m in range(M):
    Gamma[0 + index_m * T:T + index_m * T, :] = s @ np.diag(g[:, index_m])
gamma_hat_prior_CSI = np.linalg.inv(Gamma.conj().T @ Gamma) @ Gamma.conj().T @ y_tilde

C_inv_prior_CSI = np.linalg.inv(
    s @ np.diag(abs(gamma_hat_prior_CSI[:, 0]) ** 2 * lambda_k[:, 0] ** 2) @ s.T.conj() + sigma2 * np.identity(T))
print('Value of cost function just using only prior CSI: ' + str(
    utils.ML_value(gamma_hat_prior_CSI, C_inv_prior_CSI, y, s, g, M)))

## Estimator based on partial CSI and iterative ML
# Initialization thanks to prior CSI
gamma_hat_partial_CSI, C_inverse_partial_CSI = utils.algorithm(gamma_hat_prior_CSI.copy(), lambda_k, s, M, y, g, sigma2,
                                                               T, K,
                                                               iter_max=ITER_MAX)
print('Value of cost function partial CSI (prior init): ' + str(
    utils.ML_value(gamma_hat_partial_CSI, C_inverse_partial_CSI, y, s, g, M)))

## Estimator based on partial CSI and iterative ML
# Initialization without CSI
gamma_hat_partial_CSI_0_init, C_inverse_partial_CSI_0_init = utils.algorithm(np.zeros_like(gamma), lambda_k,
                                                                             s, M, y, g, sigma2, T, K,
                                                                             iter_max=ITER_MAX)
print('Value of cost function partial CSI (0 init): ' + str(
    utils.ML_value(gamma_hat_partial_CSI_0_init, C_inverse_partial_CSI_0_init, y, s, g, M)))

snr_k_partial_CSI = (np.linalg.norm(g, axis=1)**2+M*lambda_k**2)/sigma2

# Estimator based on no CSI and iterative ML (as Caire)
lambda_k = np.ones_like(lambda_k)
g = np.zeros_like(g)
gamma_hat_no_CSI, C_inverse_no_CSI = utils.algorithm(np.zeros_like(gamma), lambda_k, s, M, y, g, sigma2, T,
                                                     K, iter_max=ITER_MAX)
snr_k_no_CSI = (np.linalg.norm(g, axis=1)**2+M*lambda_k**2)/sigma2

print('Value of cost function using no CSI: ' + str(utils.ML_value(gamma_hat_no_CSI, C_inverse_no_CSI, y, s, g, M)))

plt.figure()
plt.subplot(4, 1, 1)
plt.stem(np.abs(gamma), use_line_collection=True)
plt.title('gamma')
plt.subplot(4, 1, 2)
plt.stem(np.abs(gamma_hat_prior_CSI), use_line_collection=True)
plt.title('gamma_hat_prior_CSI')
plt.subplot(4, 1, 3)
plt.stem(np.abs(gamma_hat_partial_CSI), use_line_collection=True)
plt.title('gamma_hat_partial_CSI')
# plt.subplot(5, 1, 4)
# plt.stem(np.abs(gamma_hat_partial_CSI_0_init), use_line_collection=True)
# plt.title('gamma_hat_partial_CSI_0_init')
plt.subplot(4, 1, 4)
plt.stem(np.abs(gamma_hat_no_CSI), use_line_collection=True)
plt.title('gamma_hat_no_CSI')

print(f'MSE just using prior CSI: \t{utils.MSE(gamma, gamma_hat_prior_CSI):0.2f} dB')
print(f'MSE using partial CSI:  \t{utils.MSE(gamma, gamma_hat_partial_CSI):0.2f} dB')
print(f'MSE using no CSI: \t\t\t{utils.MSE(gamma, gamma_hat_no_CSI):0.2f} dB')

plt.tight_layout()
plt.show()

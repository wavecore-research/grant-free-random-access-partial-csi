import numpy as np
import numba


def iid(dim, var=1.0) -> np.ndarray:
    """
    return iid CN(0,var) with dimensions dim
    """
    return (1 / np.sqrt(2)) * np.random.normal(size=dim, scale=np.sqrt(var)) + 1j * (1 / np.sqrt(2)) * np.random.normal(
        size=dim, scale=np.sqrt(var))


def is_illcond(arr: np.ndarray):
    """
    Check if a matrix/arr is ill conditioned or not
    :param arr:
    :return:
    """
    print(1.0 / np.linalg.cond(arr))
    # A problem with a low condition number is said to be well-conditioned,
    # while a problem with a high condition number is said to be ill-conditioned.
    if np.linalg.cond(arr) < (1.0 / np.finfo(complex).eps):
        return False
    return True


def is_diag(arr):
    i, j = arr.shape
    assert i == j
    test = arr.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])


@numba.jit(fastmath=True, nopython=True)
def prob_miss(arr, arr_est) -> float:
    arr = arr.flatten()
    arr_est = arr_est.flatten()
    num_active = np.sum(arr)
    # | A union A_hat | is equal to sum(arr AND arr_est), or only when both are 1 (active) -> 1
    num_correct = np.sum(arr_est[arr == 1])
    return 1.0 - (num_correct / num_active)


@numba.jit(fastmath=True, nopython=True)
def prob_false(arr, arr_est) -> float:
    # num inactive devices arr=0 detected as active arr_est = 1
    arr = arr.flatten()
    arr_est = arr_est.flatten()
    num_active = np.sum(arr)
    num_users = arr.size
    p_false = np.sum(arr_est[arr == 0]) / (num_users - num_active)
    return p_false


@numba.jit(fastmath=True, nopython=True)
def inv(mat):
    return np.linalg.inv(mat)


@numba.jit(fastmath=True, nopython=True)
def ZF(M: int, T: int, K: int, s: np.ndarray, g: np.ndarray, y: np.ndarray):
    MT = int(M * T)
    y_tilde = y.T.copy().reshape(MT, 1)
    Gamma = np.zeros((MT, K), dtype=np.complex_)
    for index_m in range(M):
        Gamma[0 + index_m * T:T + index_m * T, :] = s @ np.diag(g[:, index_m])
    return np.linalg.inv(Gamma.conj().T @ Gamma) @ Gamma.conj().T @ y_tilde

#
# def beta(d, model="oulu", sigma=0):
#     d = d / 1000
#
#     if model in "oulu":
#         pl0 = 128.95
#         n = 2.32
#         sigma = 7.8
#     elif model in "dortmund":
#         pl0 = 132.25
#         n = 2.65
#     elif model in "three-slope":
#         d = d * 1000  # here expressed in meters
#         if d < 10:
#             return 10 ** (-81.2 / 10)
#         elif d < 50:
#             return 10 ** ((-61.2 - 20 * np.log10(d)) / 10)
#         else:
#             return 10 ** ((-35.7 - 35 * np.log10(d) + np.random.normal(scale=8)) / 10)
#     else:
#         return ValueError
#
#     pl_db = pl0 + 10 * n * np.log10(d) + np.random.normal(scale=sigma)
#     return 10 ** (- pl_db / 10)


def generate_user_pos(num, size) -> np.ndarray:
    """
    Generates a numx2 matrix containing the x,y position per user

    :param num: number of user
    :param size: dimension of the square area (sqrt(area)=size)
    :return: a numx2 matrix containing the x,y position per user
    """
    return np.random.uniform(0, size, (num, 2))


def H(arr: np.ndarray):
    # conjugate transpose
    if arr.ndim == 1:
        arr = np.array([arr])
    return np.conj(arr.T)


@numba.jit(fastmath=True, nopython=True)
def alpha(s, C_inv, g, y_m_k_prime, lambda_k, k_prime):
    temp1 = s[:, k_prime].T.conj() @ C_inv @ s[:, k_prime]
    temp2 = y_m_k_prime.T.conj() @ C_inv @ s[:, k_prime]
    return (lambda_k[k_prime] ** 2 * np.sum(np.abs(temp2) ** 2) - temp1 * np.sum(np.abs(g[k_prime, :]) ** 2)).item()


@numba.jit(fastmath=True, nopython=True)
def beta(s, C_inv, g, y_m_k_prime, k_prime):
    return float(2 * np.abs(g[k_prime, :] @ y_m_k_prime.T.conj() @ C_inv @ s[:, k_prime]))


@numba.jit(fastmath=True, nopython=True)
def delta(s, C_inv, lambda_k, k_prime):
    return (s[:, k_prime].T.conj() @ C_inv @ s[:, k_prime] * lambda_k[k_prime] ** 2).item()


# @numba.jit(nopython=True)
# def ampl_opt(M, _alpha, _beta, _delta):
#     coef_3 = -2 * M * _delta ** 2
#     coef_2 = -_beta * _delta
#     coef_1 = (-2 * M * _delta + 2 * _alpha)
#     coef_0 = _beta
#     res = np.roots(np.array([coef_3, coef_2, coef_1, coef_0], dtype=np.complex_))
#     # check for best solution being positive and real
#     sol = -1
#     for r in res:
#         if np.isreal(r) and r >= 0:
#             sol = r if r > sol else sol
#     if sol == -1:
#         raise ValueError("No solution for poly. found.")
#     return sol


@numba.jit(nopython=True)
def ML_value(gamma_hat, C_inverse, y, s, g, M, T):
    r_out = M * np.log(np.linalg.det(C_inverse)) - M * T * np.log(np.pi) - np.trace(
        (y - s @ np.diag(gamma_hat) @ g).T.conj() @ C_inverse @ (y - s @ np.diag(gamma_hat) @ g))
    return np.real(r_out)


@numba.jit(nopython=True)
def is_realpositive(val, tol=1e-5):
    return np.imag(val) < tol and np.real(val) >= 0


@numba.jit(nopython=True, fastmath=True)
def algorithm(gamma_hat: np.ndarray, lambda_k: np.ndarray, s: np.ndarray, M: int, y: np.ndarray, g: np.ndarray,
              sigma2: float, T: int, K: int, real_gamma: np.ndarray, iter_max: int = 1000):
    iter_number = 0
    k_prime = 0
    not_converged = True

    gamma_hat = gamma_hat[:, 0].copy()
    lambda_k = lambda_k.copy()
    s = s.copy()
    y = y.copy()
    g = g.copy()

    sigma2I = sigma2 * np.identity(T)

    r = (np.abs(gamma_hat) ** 2 * lambda_k[:, 0] ** 2).astype(np.complex_)
    R = np.diag(r)
    s_H = s.T.conj()

    global_C_inverse = np.linalg.inv((s @ R @ s_H) + sigma2I)

    MSEs = np.zeros(iter_max, dtype=np.float_)

    while not_converged:
        temp = gamma_hat.copy().astype(np.complex_)
        temp[k_prime] = 0 + 0j
        y_m_k_prime = y - s @ np.diag(temp) @ g

        # if lambda_k[k_prime] == 0:
        #     gamma_hat_k_prime = (s[:, k_prime].T.conj() @ C_inverse @ (y_m_k_prime @ g[k_prime, :].T.conj())) / (
        #             s[:, k_prime].T.conj() @ C_inverse @ s[:, k_prime] * np.sum(np.abs(g[k_prime, :]) ** 2))
        # else:
        # C_minus_k_prime_inverse = np.linalg.inv(
        #     s @ np.diag(np.abs(temp) ** 2 * lambda_k[:, 0] ** 2).astype(
        #         np.complex_) @ s_H + sigma2I)

        C_inverse = global_C_inverse.copy()

        s_s_H = np.outer(s[:, k_prime], s_H[k_prime, :])
        lam_gamma = lambda_k[k_prime, :] ** 2 * np.abs(gamma_hat[k_prime] ** 2)
        up = lam_gamma * C_inverse @ s_s_H @ C_inverse
        low = 1 - lam_gamma * s_H[k_prime, :] @ C_inverse @ s[:, k_prime]

        C_minus_k_prime_inverse = C_inverse + up / low

        _alpha = alpha(s, C_minus_k_prime_inverse, g, y_m_k_prime, lambda_k, k_prime)
        _delta = delta(s, C_minus_k_prime_inverse, lambda_k, k_prime)
        _beta = beta(s, C_minus_k_prime_inverse, g, y_m_k_prime, k_prime)

        # Amplitude optimization
        coef_3 = -2 * M * _delta ** 2
        coef_2 = -_beta * _delta
        coef_1 = -2 * M * _delta + 2 * _alpha
        coef_0 = _beta
        coef = np.array([coef_3, coef_2, coef_1, coef_0], dtype=np.complex_)
        roots = np.roots(coef)
        # check for best solution being positive and real
        sol = 0.0
        found = False
        for root in roots:
            if is_realpositive(root):
                _r = np.real(root)
                sol = _r if _r >= sol else sol
                found = True
        if not found:
            print('\033[31m No solution for poly. found.')
            # raise ValueError("No solution for poly. found.")

            # do not use this value, retry next time.
            # fill with previous value
            # sol = abs(gamma_hat[k_prime])

        r_k_prime_hat = sol

        # Phase optimization
        phi_k_prime_hat = np.angle(s[:, k_prime].T.conj() @ C_inverse @ (y_m_k_prime @ g[k_prime, :].T.conj()))
        gamma_hat_k_prime = r_k_prime_hat * np.exp(1j * phi_k_prime_hat)

        # Update gamma_hat
        gamma_hat[k_prime] = gamma_hat_k_prime

        r_lam = r_k_prime_hat ** 2 * lambda_k[k_prime, 0] ** 2
        up = C_minus_k_prime_inverse @ s_s_H @ C_minus_k_prime_inverse * r_lam
        low = 1 + s_H[k_prime, :] @ C_minus_k_prime_inverse @ s[:, k_prime] * r_lam
        global_C_inverse = np.ascontiguousarray(C_minus_k_prime_inverse - up / low)

        # global_C_inverse = np.linalg.inv(
        #     s @ np.diag(np.abs(gamma_hat) ** 2 * lambda_k[:, 0] ** 2).astype(
        #         np.complex_) @ s.T.conj() + sigma2I)

        # next iteration
        k_prime = np.mod(k_prime + 1, K)

        MSEs[iter_number] = MSE(gamma_hat, real_gamma)

        # print('Iteration number: ' + str(iter_number) + ', value of cost function: '+ str(ML_value(gamma_hat)))
        iter_number += 1
        if iter_number > iter_max - 1:
            not_converged = False
    return gamma_hat.copy(), global_C_inverse.copy(), MSEs

# @numba.jit(nopython=True, fastmath=True)
def algorithm_no_csi(gamma_hat: np.ndarray, s: np.ndarray, M: int, y: np.ndarray,
              sigma2: float, T: int, K: int, real_gamma: np.ndarray, iter_max: int = 1000):
    lambda_k = np.ones((K, 1))

    iter_number = 0
    k_prime = 0
    not_converged = True

    gamma_hat = gamma_hat[:, 0].copy()
    lambda_k = lambda_k.copy()
    s = s.copy()
    y = y.copy()
    g = np.zeros((K, M), dtype=np.complex_)

    sigma2I = sigma2 * np.identity(T)

    r = (np.abs(gamma_hat) ** 2 * lambda_k[:, 0] ** 2).astype(np.complex_)
    R = np.diag(r)
    s_H = s.T.conj()

    global_C_inverse = np.linalg.inv((s @ R @ s_H) + sigma2I)

    MSEs = np.zeros(iter_max, dtype=np.float_)

    while not_converged:
        temp = gamma_hat.copy().astype(np.complex_)
        temp[k_prime] = 0 + 0j
        y_m_k_prime = y - s @ np.diag(temp) @ g

        # if lambda_k[k_prime] == 0:
        #     gamma_hat_k_prime = (s[:, k_prime].T.conj() @ C_inverse @ (y_m_k_prime @ g[k_prime, :].T.conj())) / (
        #             s[:, k_prime].T.conj() @ C_inverse @ s[:, k_prime] * np.sum(np.abs(g[k_prime, :]) ** 2))
        # else:
        # C_minus_k_prime_inverse = np.linalg.inv(
        #     s @ np.diag(np.abs(temp) ** 2 * lambda_k[:, 0] ** 2).astype(
        #         np.complex_) @ s_H + sigma2I)

        C_inverse = global_C_inverse.copy()

        s_s_H = np.outer(s[:, k_prime], s_H[k_prime, :])
        lam_gamma = lambda_k[k_prime, :] ** 2 * np.abs(gamma_hat[k_prime] ** 2)
        up = lam_gamma * C_inverse @ s_s_H @ C_inverse
        low = 1 - lam_gamma * s_H[k_prime, :] @ C_inverse @ s[:, k_prime]

        C_minus_k_prime_inverse = C_inverse + up / low

        _alpha = alpha(s, C_minus_k_prime_inverse, g, y_m_k_prime, lambda_k, k_prime)
        _delta = delta(s, C_minus_k_prime_inverse, lambda_k, k_prime)
        #_beta = beta(s, C_minus_k_prime_inverse, g, y_m_k_prime, k_prime)

        # Amplitude optimization
        r_k_prime_hat = np.sqrt((_alpha - M*_delta)/(M*_delta**2))

        if np.imag(r_k_prime_hat) > 1e-5: # if imaginary apart is greater than a tolerance, we expect it to be imagniary, not real
            r_k_prime_hat = 0

        # Update gamma_hat
        gamma_hat[k_prime] = r_k_prime_hat

        r_lam = r_k_prime_hat ** 2 * lambda_k[k_prime, 0] ** 2
        up = C_minus_k_prime_inverse @ s_s_H @ C_minus_k_prime_inverse * r_lam
        low = 1 + s_H[k_prime, :] @ C_minus_k_prime_inverse @ s[:, k_prime] * r_lam
        global_C_inverse = np.ascontiguousarray(C_minus_k_prime_inverse - up / low)

        # global_C_inverse = np.linalg.inv(
        #     s @ np.diag(np.abs(gamma_hat) ** 2 * lambda_k[:, 0] ** 2).astype(
        #         np.complex_) @ s.T.conj() + sigma2I)

        # next iteration
        k_prime = np.mod(k_prime + 1, K)

        MSEs[iter_number] = MSE(gamma_hat, real_gamma)

        # print('Iteration number: ' + str(iter_number) + ', value of cost function: '+ str(ML_value(gamma_hat)))
        iter_number += 1
        if iter_number > iter_max - 1:
            not_converged = False
    return gamma_hat.copy(), global_C_inverse.copy(), MSEs

@numba.jit(nopython=True, fastmath=True)
def MSE(mat: np.ndarray, est: np.ndarray):
    return np.average(np.abs(np.abs(mat.flatten()) - np.abs(est.flatten())) ** 2)


def MSE_dB(mat: np.ndarray, est: np.ndarray):
    return 10 * np.log10(MSE(mat, est))


@numba.jit(nopython=True, fastmath=True)
def SINR(arr, est):
    I_err = np.abs(np.imag(arr) - np.imag(est))
    Q_err = np.abs(np.real(arr) - np.real(est))
    return np.average(np.abs(arr) ** 2 / (I_err ** 2 + Q_err ** 2), axis=-1)


def SINR_dB(arr, est):
    return 10 * np.log10(SINR(arr, est))


@numba.jit(nopython=True, fastmath=True)
def sum_rate(arr, est):
    return np.sum(np.log2(np.ones(arr.shape[0]) + SINR(arr, est)))

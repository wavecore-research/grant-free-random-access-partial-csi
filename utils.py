import numpy as np
import numba
from numba import prange
import cupy as cp


def iid(dim, var=1.0) -> np.ndarray:
    """
    return iid CN(0,var) with dimensions dim
    """
    return np.sqrt(var/2) * (np.random.normal(size=dim, scale=1) + 1j * np.random.normal(
        size=dim, scale=1))


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


@numba.jit(fastmath=True, nopython=True, nogil=True)
def prob_miss(arr, arr_est) -> float:
    arr = arr.flatten()
    arr_est = arr_est.flatten()
    num_active = np.sum(arr)
    # | A union A_hat | is equal to sum(arr AND arr_est), or only when both are 1 (active) -> 1
    num_correct = np.sum(arr_est[arr == 1])
    return 1.0 - (num_correct / num_active)


@numba.jit(fastmath=True, nopython=True, nogil=True)
def prob_false(arr, arr_est) -> float:
    # num inactive devices arr=0 detected as active arr_est = 1
    arr = arr.flatten()
    arr_est = arr_est.flatten()
    num_active = np.sum(arr)
    num_users = arr.size
    p_false = np.sum(arr_est[arr == 0]) / (num_users - num_active)
    return p_false


# @numba.jit(fastmath=True, nopython=True)
# def inv(mat):
#     return np.linalg.inv(mat)


def Gamma_cp(M: int, s: cp.ndarray, g: cp.ndarray):
    return cp.vstack([(s @ cp.diag(g[:, index_m])) for index_m in range(M)])


# old gamma function, above is faster
# @numba.jit(fastmath=True, nopython=True, parallel=True)
# def Gamma(M: int, T: int, K: int, s: np.ndarray, g: np.ndarray):
#     _Gamma = np.empty((int(M * T), K), dtype=np.complex_)
#     for index_m in prange(M):
#         _Gamma[index_m * T:T + index_m * T, :] = s @ np.diag(g[:, index_m])
#     return _Gamma


# @numba.jit(fastmath=True, nopython=True, parallel=True)
# def MF(M: int, T: int, K: int, s: np.ndarray, g: np.ndarray, y: np.ndarray, eps_a: float, p_tx: float, sigma2: float):
#     MT = int(M * T)
#     y_tilde = y.T.copy().reshape(MT, 1)
#     _Gamma = Gamma(M, T, K, s, g)
#     Gamma_diag = _Gamma.conj().T @ _Gamma
#
#     return np.diag((1 / np.diag(Gamma_diag) + (sigma2 / (p_tx * eps_a)))) @ _Gamma.conj().T @ y_tilde

# removed ZF for RZF
# def ZF(M: int, T: int, K: int, s: np.ndarray, g: np.ndarray, y: np.ndarray):
#     MT = int(M * T)
#     y_tilde = y.T.copy().reshape(MT, 1)
#
#     _Gamma = Gamma_cp(M, cp.asarray(s), cp.asarray(g))
#
#     _Gamma_H = cp.linalg.inv(cp.matmul(_Gamma.conj().T, _Gamma))
#     _Gamma_pinv = cp.matmul(_Gamma_H, _Gamma.conj().T)
#
#     return cp.matmul(_Gamma_pinv, cp.asarray(y_tilde)).get()


def RZF(M: int, T: int, K: int, s: np.ndarray, g: np.ndarray, y: np.ndarray, sigma2: float, eps_a: float, p_tx: float):
    MT = int(M * T)
    y_tilde = y.T.copy().reshape(MT, 1)

    _Gamma = Gamma_cp(M, cp.asarray(s), cp.asarray(g))

    _D_inv = (1 / (eps_a * p_tx)) * cp.identity(K)

    _Gamma_H = cp.linalg.inv(cp.matmul(_Gamma.conj().T, _Gamma) + sigma2 * _D_inv)
    _Gamma_pinv = cp.matmul(_Gamma_H, _Gamma.conj().T)

    return cp.matmul(_Gamma_pinv, cp.asarray(y_tilde)).get()


# @numba.jit(fastmath=True, nopython=True, parallel=True)
# def ZF(M: int, T: int, K: int, s: np.ndarray, g: np.ndarray, y: np.ndarray):
#     MT = int(M * T)
#     y_tilde = y.T.copy().reshape(MT, 1)
#     _Gamma = Gamma(M, T, K, s, g)
#
#     return inv(_Gamma.conj().T @ _Gamma) @ _Gamma.conj().T @ y_tilde


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


@numba.jit(nopython=True, cache=True)
def alpha(s, C_inv, g, y_m_k_prime, lambda_k, k_prime):
    temp1 = s[:, k_prime].T.conj() @ C_inv @ s[:, k_prime]
    temp2 = y_m_k_prime.T.conj() @ C_inv @ s[:, k_prime]
    return (lambda_k[k_prime] ** 2 * np.sum(np.abs(temp2) ** 2) - temp1 * np.sum(np.abs(g[k_prime, :]) ** 2)).item()
    # np.outer(np.conjugate(y_m_k_prime).T @ C_inv, s[:, k_prime]) ** 2


@numba.jit(nopython=True, cache=True)
def beta(s, C_inv, g, y_m_k_prime, k_prime):
    return float(2 * np.abs(g[k_prime, :] @ y_m_k_prime.T.conj() @ C_inv @ s[:, k_prime]))


@numba.jit(nopython=True, cache=True)
def delta(s, C_inv, lambda_k, k_prime):
    return np.real((s[:, k_prime].T.conj() @ C_inv @ s[:, k_prime] * lambda_k[k_prime] ** 2).item())


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

#
# @numba.jit(nopython=True)
# def LL_value(gamma_hat, C_inverse, y, s, g, M, T):
# r_out = M * np.log(np.linalg.det(np.linalg.inv(C_inverse))) - M * T * np.log(np.pi) - np.trace(
#     (y - s @ np.diag(gamma_hat) @ g).T.conj() @ C_inverse @ (y - s @ np.diag(gamma_hat) @ g))
# return np.real(r_out)

@numba.jit(nopython=True, fastmath=True, cache=True)
def LL(gamma, g, s, C_inv, M, T, K, y):
    _det = np.linalg.det(inv(C_inv))
    ll1 = - M * np.log(_det) - M * T * np.log(np.pi)
    ll2 = 0
    for m in range(M):
        gsk = np.zeros(T, dtype=np.complex_)
        for k in range(K):
            gsk += g[k, m] * s[:, k] * gamma[k]
        ygsk = y[:, m] - gsk
        ll2 -= ygsk.conj().T @ C_inv @ ygsk

    return np.real(ll1 + ll2)


# def LL_value(C_inverse, _alpha, _beta, _delta, r, M):
#     _det = np.linalg.det(inv(C_inverse))
#     return np.real(
#         - M * np.log(_det) + ((_alpha * r ** 2 + _beta * r) / (1 + _delta * r ** 2)))

@numba.jit(nopython=True, fastmath=True, cache=True)
def f_rk_prime(M, C_minus_k_prime_inverse, _alpha, _beta, _delta, r_k_prime_hat, s, s_H, k_prime, s_s_H, lambda_k,
               gamma_hat, g, T, K, y, phi_k_prime_hat, current_ll):
    print("LL not monotonically increasing...")
    import matplotlib.pyplot as plt
    fig, ax_ll = plt.subplots()
    ax_der = ax_ll.twinx()
    ax3 = ax_der.twiny()
    ax_ll.get_shared_x_axes().join(ax_der, ax3)

    r_range, delta_x = np.linspace(0, 3, num=1000, retstep=True)

    def C_r(r):
        r_lam = r ** 2 * lambda_k[k_prime, 0] ** 2
        up = C_minus_k_prime_inverse @ s_s_H @ C_minus_k_prime_inverse * r_lam
        low = 1 + s_H[k_prime, :] @ C_minus_k_prime_inverse @ s[:, k_prime] * r_lam

        return inv(C_minus_k_prime_inverse - up / low)

    def fr_fun(r):  # (13)
        _C = C_r(r)
        fr1 = - M * np.log(np.linalg.det(_C))
        fr2 = (_alpha * r ** 2 + _beta * r) / (1 + _delta * r ** 2)
        fr = fr1 + fr2
        return fr, fr1, fr2

    def der_fr_fun(r):  # (15)
        denom = (1 + _delta * (r ** 2))
        der_fr1 = (-2 * M) * ((_delta * r) / denom)
        der_fr2 = (-(r ** 2) * _beta * _delta + r * 2 * _alpha + _beta) / (denom ** 2)
        return (der_fr1 + der_fr2), der_fr1, der_fr2

    def derivitive(y, dx):
        return np.gradient(y, dx)



    fr_k_prime, fr1_k_prime, fr2_k_prime = fr_fun(r_k_prime_hat)
    der_fr, der_fr1, der_fr2 = zip(*[der_fr_fun(r) for r in r_range])

    fr_range, fr1_range, fr2_range = zip(*[fr_fun(r) for r in r_range])
    der_range = derivitive(fr_range, delta_x)

    ax_der.plot(r_range, np.real(der_range), ls="--", label="df computed")
    ax_der.plot(r_range, np.real(der_fr), ls="--", label="df analytical")
    # plt.plot(r_range, np.real(der_fr2), ls="-.", color=plt.gca().lines[-1].get_color(),
    #          label="df analytical term 2 (real part)")

    # plt.plot(r_range, np.real(derivitive(fr_range, delta_x)), ls="--", label="df computed term 1 (real part)")
    # plt.plot(r_range, np.real(derivitive(fr2_range, delta_x)), ls="-.", color=plt.gca().lines[-1].get_color(),
    #          label="df computed term 2 (real part)")

    #
    ax_ll.scatter(r_k_prime_hat, fr_k_prime, color="green")
    ax_der.scatter(r_k_prime_hat, der_fr_fun(r_k_prime_hat)[0], color="green")
    #
    #
    res = np.real(
        [- M * np.log(np.linalg.det(C_r(r))) + ((_alpha * r ** 2 + _beta * r) / (1 + _delta * r ** 2)) for r in
         r_range])
    ax_ll.plot(r_range, res, label="LL computed")
    #
    # der_computed = [(res[i + 1] - res[i]) / delta_x for i in range(10000 - 1)]
    # ax_der.plot(r_range[:-1], der_computed, color="blue", ls="--")
    #
    opt_idx = np.argmin(np.abs(der_range))
    r_k_opt = r_range[opt_idx]
    ax_der.scatter(r_k_opt, der_range[opt_idx], color="orange", ls="--")
    #
    r_k_opt = r_range[np.argmax(fr_range)]
    ax_ll.scatter(r_k_opt, np.max(fr_range), color="orange")
    #
    # der_computed = derivitive(res, step=delta_x)
    # ax_der.plot(r_range[:-1], der_computed, color="blue", ls="--")
    #
    # if not np.isclose(der, 0):
    #     print("Wrong der!")
    #
    # ax_der.scatter(r_k_prime_hat, der, color="green")
    #
    # denom = (1 + _delta * r_range ** 2)
    # der = np.real(((-2 * M * _delta * r_range) / denom) + (
    #         -r_range ** 2 * _beta * _delta + r_range * 2 * _alpha + _beta) / denom ** 2)
    #
    # ax_der.plot(r_range, der, color='red')
    #
    ax_ll.legend()
    ax_der.legend()
    plt.show()

    plt.figure()
    ll_range = np.zeros(len(r_range))
    for i_r, r in enumerate(r_range):
        gamma_hat_r = gamma_hat.copy()
        gamma_hat_r[k_prime] = r * np.exp(1j * phi_k_prime_hat)
        ll_range[i_r] = LL(gamma_hat_r, g, s, inv(C_r(r)), M, T, K, y)

    plt.plot(r_range, np.real(ll_range), label="Total LL over r")
    plt.scatter(r_k_prime_hat, current_ll)
    if current_ll != LL(gamma_hat, g, s, inv(C_r(r_k_prime_hat)), M, T, K, y):
        print("Error")
    plt.legend()
    plt.show()


@numba.jit(nopython=True, fastmath=True, cache=True)
def is_realpositive(val, tol=1e-5):
    return np.imag(val) < tol and np.real(val) >= 0

@numba.jit(nopython=True, fastmath=True, cache=True)
def is_full_rank(A):
    return np.linalg.matrix_rank(A) >= np.min(np.array(list(A.shape))) # weird casting from tuple->list->np.ndarray bcs of numba

@numba.jit(nopython=True, fastmath=True, cache=True)
def C(_gamma_hat, _lambda_k, _s, _sigma2I):
    _r = (np.abs(_gamma_hat) ** 2 * _lambda_k[:, 0] ** 2).astype(np.complex_)
    _R = np.diag(_r)
    _s_H = _s.T.conj()
    _C = (_s @ _R @ _s_H) + _sigma2I
    return _C


@numba.jit(nopython=True, fastmath=True, cache=True)
def inv(A):
    # if not is_full_rank(A):
    #     print("Matrix is not full rank, will not be abel to inverse")
    return np.linalg.inv(A)

@numba.jit(nopython=True, fastmath=True, cache=True)
def algorithm(gamma_hat: np.ndarray, lambda_k: np.ndarray, s: np.ndarray, M: int, y: np.ndarray, g: np.ndarray,
              sigma2: float, T: int, K: int, real_gamma: np.ndarray, iter_max: int = 1000):
    iter_number = 0
    k_prime = 0
    not_converged = True

    gamma_hat = gamma_hat[:, 0].copy()

    sigma2I = sigma2 * np.identity(T)

    s_H = s.T.conj()

    global_C_inverse = inv(C(gamma_hat, lambda_k, s, sigma2I))

    # global_C_inverse = np.linalg.inv(np.sum([(lambda_k[k, 0] * r[k] * np.outer(s[:, k], s_H[k, :])) for k in range(K)]) + sigma2I)
    #
    # if not np.isclose(global_C_inverse, global_C_inverse_1).all():
    #     print("WRONG C_inv")

    MSEs = np.zeros(iter_max, dtype=np.float_)
    LLs = np.zeros(iter_max, dtype=np.float_)

    s_s_H = [np.zeros((K, K), dtype=np.complex128)] * K
    for k in prange(K):
        s_s_H[k] = np.outer(s[:, k], s_H[k, :])

    while not_converged:
        temp = gamma_hat.copy().astype(np.complex_)
        temp[k_prime] = 0 + 0j
        y_m_k_prime = y - s @ np.diag(temp) @ g
        # y_m_k_prime = np.array(
        #     [y[:, m] - np.sum([gamma_hat[k] * s[:, k] if k != k_prime else 0 + 0j for k in range(K)] * g[:, m]) for m
        #      in range(M)]).T

        # if np.isclose(y_m_k_prime_2, y_m_k_prime.T).all():
        #     print("y_m_k not cirrect :'(")

        # Phase optimization
        C_inverse = global_C_inverse.copy()
        phi_k_prime_hat = np.angle(s[:, k_prime].T.conj() @ C_inverse @ (y_m_k_prime @ g[k_prime, :].T.conj()))

        # Compute C_minus_k_prime_inverse
        lam_gamma = lambda_k[k_prime, 0] ** 2 * np.abs(gamma_hat[k_prime] ** 2)
        up = lam_gamma * C_inverse @ s_s_H[k_prime] @ C_inverse
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
        # roots = np.roots(coef)
        roots = np.roots(coef)
        # check for best solution being positive and real
        sol = 0.0
        found = False
        for root in roots:
            if is_realpositive(root):
                if found:
                    print('\033[31m More than one root found')
                _r = np.real(root)
                sol = _r if _r >= sol else sol
                found = True
        if not found:
            print('\033[31m No solution for poly. found.')

        r_k_prime_hat = sol

        MSEs[iter_number] = MSE(gamma_hat, real_gamma)

        gamma_hat_k_prime = r_k_prime_hat * np.exp(1j * phi_k_prime_hat)

        # Update gamma_hat
        gamma_hat[k_prime] = gamma_hat_k_prime

        r_lam = r_k_prime_hat ** 2 * lambda_k[k_prime].item() ** 2
        up = C_minus_k_prime_inverse @ s_s_H[k_prime] @ C_minus_k_prime_inverse * r_lam
        low = 1 + s_H[k_prime, :] @ C_minus_k_prime_inverse @ s[:, k_prime] * r_lam
        global_C_inverse = np.ascontiguousarray(C_minus_k_prime_inverse - up / low)

        # global_C_inverse_2 = np.linalg.inv(C(gamma_hat, lambda_k, s, sigma2I))
        #
        # if not np.isclose(global_C_inverse_2, global_C_inverse).all():
        #     print("Computed C is not correct")

        LLs[iter_number] = LL(gamma_hat, g, s, global_C_inverse, M, T, K, y)
        if (LLs[iter_number] < LLs[iter_number - 1]) and iter_number > 0:
            print("LL not monotonically increasing...")
            # f_rk_prime(M, C_minus_k_prime_inverse, _alpha, _beta, _delta, r_k_prime_hat, s, s_H, k_prime,
            #            s_s_H[k_prime], lambda_k, gamma_hat, g, T, K, y, phi_k_prime_hat, LLs[iter_number])

        # global_C_inverse = np.linalg.inv(
        #     s @ np.diag(np.abs(gamma_hat) ** 2 * lambda_k[:, 0] ** 2).astype(
        #         np.complex_) @ s.T.conj() + sigma2I)

        # next iteration
        k_prime = np.mod(k_prime + 1, K)

        # print('Iteration number: ' + str(iter_number) + ', value of cost function: '+ str(LLs[iter_number].item()))
        iter_number += 1

        if iter_number > iter_max - 1:
            not_converged = False

    return gamma_hat.copy(), global_C_inverse.copy(), (MSEs, LLs)


@numba.jit(nopython=True, fastmath=True, cache=True)
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
    s_s_H = [np.zeros((K, K), dtype=np.complex128)] * K
    for k in range(K):
        s_s_H[k] = np.outer(s[:, k], s_H[k, :])

    #  MSEs = np.zeros(iter_max, dtype=np.float_)

    while not_converged:
        temp = gamma_hat.copy().astype(np.complex_)
        temp[k_prime] = 0 + 0j
        y_m_k_prime = y - s @ np.diag(temp) @ g

        C_inverse = global_C_inverse.copy()

        lam_gamma = lambda_k[k_prime, :] ** 2 * np.abs(gamma_hat[k_prime] ** 2)
        up = lam_gamma * C_inverse @ s_s_H[k_prime] @ C_inverse
        low = 1 - lam_gamma * s_H[k_prime, :] @ C_inverse @ s[:, k_prime]

        C_minus_k_prime_inverse = C_inverse + up / low

        _alpha = alpha(s, C_minus_k_prime_inverse, g, y_m_k_prime, lambda_k, k_prime)
        _delta = delta(s, C_minus_k_prime_inverse, lambda_k, k_prime)
        # _beta = beta(s, C_minus_k_prime_inverse, g, y_m_k_prime, k_prime)

        # Amplitude optimization
        r_k_prime_hat = np.sqrt((_alpha - M * _delta) / (M * _delta ** 2))

        if np.imag(
                r_k_prime_hat) > 1e-5:  # if imaginary apart is greater than a tolerance, we expect it to be imagniary, not real
            r_k_prime_hat = 0

        # Update gamma_hat
        gamma_hat[k_prime] = r_k_prime_hat

        r_lam = r_k_prime_hat ** 2 * lambda_k[k_prime, 0] ** 2
        up = C_minus_k_prime_inverse @ s_s_H[k_prime] @ C_minus_k_prime_inverse * r_lam
        low = 1 + s_H[k_prime, :] @ C_minus_k_prime_inverse @ s[:, k_prime] * r_lam
        global_C_inverse = np.ascontiguousarray(C_minus_k_prime_inverse - up / low)

        # next iteration
        k_prime = np.mod(k_prime + 1, K)

        # MSEs[iter_number] = MSE(gamma_hat, real_gamma)

        # print('Iteration number: ' + str(iter_number) + ', value of cost function: '+ str(ML_value(gamma_hat)))
        iter_number += 1
        if iter_number > iter_max - 1:
            not_converged = False
    return gamma_hat.copy(), global_C_inverse.copy(), None  # MSEs


#
@numba.jit(nopython=True, fastmath=True)
def MSE(mat: np.ndarray, est: np.ndarray):
    return np.mean(np.abs(np.abs(mat.flatten()) - np.abs(est.flatten())) ** 2)


#
# def MSE_dB(mat: np.ndarray, est: np.ndarray):
#     return 10 * np.log10(MSE(mat, est))


@numba.jit(nopython=True, fastmath=True, cache=True)
def SINR(arr, est):
    I_err = np.abs(np.imag(arr) - np.imag(est))
    Q_err = np.abs(np.real(arr) - np.real(est))
    return np.average(np.abs(arr) ** 2 / (I_err ** 2 + Q_err ** 2), axis=-1)


def SINR_dB(arr, est):
    return 10 * np.log10(SINR(arr, est))


@numba.jit(nopython=True, fastmath=True, cache=True)
def sum_rate(arr, est):
    return np.sum(np.log2(np.ones(arr.shape[0]) + SINR(arr, est)))


def data_from_file(f):
    data = np.load(f, allow_pickle=True)
    params = data["params"].item()
    return dict(data), params


def is_same_params(p1: dict, p2: dict):
    # params = {
    #     "lambdas": lambdas,
    #     "preamble_lengths":preamble_lengths,
    #     "snrs_dB":snrs_dB,
    #     "antennas":antennas,
    #     "users":users
    # }

    is_same = True
    for key in p1.keys():
        if len(p1[key]) != len(p2[key]):
            is_same = False
            break

        if not np.allclose(np.asarray(p1[key]), np.asarray(p2[key])):
            is_same = False
            break
    return is_same


def merge(d1, d2):
    if not (np.asarray(d1["SHAPE_PROB"]) == np.asarray(d2["SHAPE_PROB"])).all():
        return None

    d1["pa_prior_csi"] = (d1["pa_prior_csi"] + d2["pa_prior_csi"]) / 2

    d1["md_prior_csi"] = (d1["md_prior_csi"] + d2["md_prior_csi"]) / 2
    d1["pa_partial_csi_ZF"] = (d1["pa_partial_csi_ZF"] + d2["pa_partial_csi_ZF"]) / 2
    d1["md_partial_csi_ZF"] = (d1["md_partial_csi_ZF"] + d2["md_partial_csi_ZF"]) / 2
    d1["pa_partial_csi"] = (d1["pa_partial_csi"] + d2["pa_partial_csi"]) / 2
    d1["md_partial_csi"] = (d1["md_partial_csi"] + d2["md_partial_csi"]) / 2
    d1["pa_prior_csi"] = (d1["pa_prior_csi"] + d2["pa_prior_csi"]) / 2

    if "pa_no_csi" in d1.keys() and "pa_no_csi" in d2.keys():
        d1["pa_no_csi"] = (d1["pa_no_csi"] + d2["pa_no_csi"]) / 2
        d1["md_no_csi"] = (d1["md_no_csi"] + d2["md_no_csi"]) / 2

    return d1

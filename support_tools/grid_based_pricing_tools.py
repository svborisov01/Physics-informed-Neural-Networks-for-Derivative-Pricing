import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def heston_fd_european(
    option_type="call",
    S_max=200.0,
    v_max=1.0,
    T=1.0,
    K=100.0,
    r=0.05,
    kappa=2.0,
    theta=0.04,
    sigma=0.3,
    rho=-0.7,
    N_S=200,
    N_v=50,
    N_t=200,
):


    is_call = option_type.lower() == "call"

    dS = S_max / N_S
    dv = v_max / N_v
    dt = T / N_t

    S = np.linspace(0.0, S_max, N_S + 1)
    v = np.linspace(0.0, v_max, N_v + 1)
    t = np.linspace(0.0, T, N_t + 1)  # t[0]=0, t[-1]=T

    def idx(i, j):
        return j * (N_S + 1) + i

    N = (N_S + 1) * (N_v + 1)

    # allocate full tensor: time × v × S
    U_all = np.zeros((N_t + 1, N_v + 1, N_S + 1))

    # terminal payoff at t = T  -> index N_t
    for j in range(N_v + 1):
        for i in range(N_S + 1):
            if is_call:
                payoff = max(S[i] - K, 0.0)
            else:
                payoff = max(K - S[i], 0.0)
            U_all[N_t, j, i] = payoff

    # flatten current time layer for linear solves
    U_flat = U_all[N_t].ravel()

    # build operator L exactly as before
    L = lil_matrix((N, N))
    for j in range(1, N_v):
        vj = v[j]
        for i in range(1, N_S):
            Si = S[i]
            k = idx(i, j)

            a_SS = 0.5 * vj * Si * Si
            a_Sv = rho * sigma * vj * Si
            a_vv = 0.5 * sigma * sigma * vj
            a_S  = r * Si
            a_v  = kappa * (theta - vj)
            a_0  = -r

            L[k, idx(i, j)] += (
                a_SS * (-2.0 / dS**2)
                + a_vv * (-2.0 / dv**2)
                + a_0
            )
            L[k, idx(i + 1, j)] += (
                a_SS * (1.0 / dS**2)
                + a_S * (1.0 / (2.0 * dS))
            )
            L[k, idx(i - 1, j)] += (
                a_SS * (1.0 / dS**2)
                - a_S * (1.0 / (2.0 * dS))
            )
            L[k, idx(i, j + 1)] += (
                a_vv * (1.0 / dv**2)
                + a_v * (1.0 / (2.0 * dv))
            )
            L[k, idx(i, j - 1)] += (
                a_vv * (1.0 / dv**2)
                - a_v * (1.0 / (2.0 * dv))
            )
            coeff_mixed = a_Sv / (4.0 * dS * dv)
            L[k, idx(i + 1, j + 1)] += +coeff_mixed
            L[k, idx(i + 1, j - 1)] += -coeff_mixed
            L[k, idx(i - 1, j + 1)] += -coeff_mixed
            L[k, idx(i - 1, j - 1)] += +coeff_mixed

    # boundaries: same pattern
    for j in range(N_v + 1):
        k = idx(0, j)
        L[k, :] = 0.0
        L[k, k] = 1.0
        k = idx(N_S, j)
        L[k, :] = 0.0
        L[k, k] = 1.0

    for i in range(1, N_S):
        k0 = idx(i, 0)
        kN = idx(i, N_v)
        L[k0, :] = 0.0
        L[k0, k0] = 1.0
        L[kN, :] = 0.0
        L[kN, kN] = 1.0

    L = csr_matrix(L)
    I = csr_matrix(np.eye(N))
    A = (I - dt * L)

    # backward in time; store each slice
    for n in range(N_t - 1, -1, -1):
        tn = t[n]
        tau = T - tn

        b = U_flat.copy()

        # impose boundaries in RHS
        if is_call:
            val_S0 = 0.0
        else:
            val_S0 = K * np.exp(-r * tau)
        if is_call:
            val_Smax = S_max - K * np.exp(-r * tau)
            val_Smax = max(val_Smax, 0.0)
        else:
            val_Smax = 0.0

        for j in range(N_v + 1):
            b[idx(0, j)]    = val_S0
            b[idx(N_S, j)]  = val_Smax

        for i in range(1, N_S):
            b[idx(i, 0)]   = U_flat[idx(i, 1)]
            b[idx(i, N_v)] = U_flat[idx(i, N_v - 1)]

        U_flat = spsolve(A, b)
        U_all[n] = U_flat.reshape(N_v + 1, N_S + 1)

    return t, S, v, U_all
"""
Accuracy tests for trained PINN models.

The functions below delegate to the universal wrapper in model_wrapper.py.
They are kept for backward compatibility with existing notebooks.
"""

from support_tools.model_wrapper import run_slice_test


def heston_2d_slice_test(
    pinn,
    v=0.3,
    r=0.2,
    kappa=2.0,
    theta=0.3,
    sigma=0.2,
    rho=0.7,
    K=150,
    sigma_mode="mean_reverting",
    return_values=False,
):
    """Test Heston PINN accuracy against COS-method Heston pricing."""
    return run_slice_test(
        pinn,
        return_values=return_values,
        v=v,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        sigma_mode=sigma_mode,
        K=K,
        tau_min=0.2,
        tau_max=4.0,
        n_tau=50,
        s_min=50.0,
        s_max=700.0,
        s_step=10.0,
    )


def bs_2d_slice_test(
    pinn,
    r=0.02,
    sigma_bs=0.2,
    K=1.0,
    tau_min=0.05,
    tau_max=2.0,
    n_tau=50,
    s_min=0.2,
    s_max=3.0,
    s_step=0.05,
    return_values=False,
):
    """Test PINN accuracy against Black–Scholes for fixed r and sigma."""
    return run_slice_test(
        pinn,
        return_values=return_values,
        r=r,
        sigma=sigma_bs,
        K=K,
        tau_min=tau_min,
        tau_max=tau_max,
        n_tau=n_tau,
        s_min=s_min,
        s_max=s_max,
        s_step=s_step,
    )

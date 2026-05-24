import numpy as np
import torch
from support_tools.analytical_pricing_tools import heston_call_price, heston_put_price, bs_option_torch

def heston_2d_slice_test(pinn, v=0.3, r=0.2, kappa=2.0, theta=0.3, sigma=0.2, rho=0.7, K=150, return_values = False):
    res = []
    pinn_prices = []
    anal_prices = []

    tau_grid = np.linspace(0.2, 4.0, 50)
    spot_grid = np.arange(50, 700, 10)

    for spot in spot_grid:
        row_mse = []
        row_pinn = []
        row_anal = []

        for tau_iter in tau_grid:
            price_pinn = pinn.predict_price(
                S=spot,
                K=K,
                v=v,
                tau=tau_iter,
                r=r,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                rho=rho,
                sigma_mode="mean_reverting",
            )

            price_pinn = float(price_pinn.detach().cpu().reshape(-1)[0])

            if pinn.call_put.lower() == "call":
                price_analytical = heston_call_price(
                    spot, K=K, T=tau_iter, r=r, q=0.0,
                    v0=v, kappa=kappa, theta=theta,
                    sigma=sigma, rho=rho
                )[0]
            elif pinn.call_put.lower() == "put":
                price_analytical = heston_put_price(
                    spot, K=K, T=tau_iter, r=r, q=0.0,
                    v0=v, kappa=kappa, theta=theta,
                    sigma=sigma, rho=rho, verbose = False
                )[0]
            else:
                raise ValueError("pinn.call_put must be either 'Call' or 'Put'")

            mse = (price_pinn - price_analytical) ** 2

            row_pinn.append(price_pinn)
            row_anal.append(price_analytical)
            row_mse.append(mse)

        pinn_prices.append(row_pinn)
        anal_prices.append(row_anal)
        res.append(row_mse)

    res = np.array(res)
    pinn_prices = np.array(pinn_prices)
    anal_prices = np.array(anal_prices)

    print("MSE =", np.mean(res))
    print("RMSE =", np.sqrt(np.mean(res)))
    print("MAE =", np.mean(np.abs(pinn_prices - anal_prices)))

    if return_values:
        return spot_grid, tau_grid, res, pinn_prices, anal_prices

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
    """
    Test PINN accuracy against Black–Scholes for fixed r and sigma_bs.

    - Grids over (S, tau).
    - Uses bs_option_torch as ground truth.
    - Prints MSE, RMSE, MAE.
    - Option type is taken from pinn.call_put.
    """

    res = []
    pinn_prices = []
    anal_prices = []

    # Time-to-maturity grid and spot grid (similar spirit to Heston test)
    tau_grid = np.linspace(tau_min, tau_max, n_tau)
    spot_grid = np.arange(s_min, s_max + 1e-8, s_step)

    call_put = pinn.call_put if hasattr(pinn, "call_put") else "Call"

    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    for spot in spot_grid:
        row_mse = []
        row_pinn = []
        row_anal = []

        for tau_iter in tau_grid:
            if hasattr(pinn, "sigma"):
                price_pinn_t = pinn.predict_price(
                S=spot,
                K=K,
                tau=tau_iter,
            )
            else:
            # PINN price (normalized architecture: predict_price multiplies by K)
                price_pinn_t = pinn.predict_price(
                    S=spot,
                    K=K,
                    tau=tau_iter,
                    r=r,
                    sigma=sigma_bs,
                )

            price_pinn = float(price_pinn_t.detach().to("cpu").reshape(-1)[0])

            # Analytical BS price
            S_t = torch.tensor([spot], device=device, dtype=dtype)
            K_t = torch.tensor([K], device=device, dtype=dtype)
            tau_t = torch.tensor([tau_iter], device=device, dtype=dtype)
            r_t = torch.tensor([r], device=device, dtype=dtype)
            sigma_t = torch.tensor([sigma_bs], device=device, dtype=dtype)

            price_analytical_t = bs_option_torch(
                S=S_t,
                K=K_t,
                tau=tau_t,
                r=r_t,
                sigma_bs=sigma_t,
                call_put=call_put,
            )
            price_analytical = float(price_analytical_t.detach().to("cpu").reshape(-1)[0])

            mse = (price_pinn - price_analytical) ** 2

            row_pinn.append(price_pinn)
            row_anal.append(price_analytical)
            row_mse.append(mse)

        pinn_prices.append(row_pinn)
        anal_prices.append(row_anal)
        res.append(row_mse)

    res = np.array(res)
    pinn_prices = np.array(pinn_prices)
    anal_prices = np.array(anal_prices)

    print("MSE =", np.mean(res))
    print("RMSE =", np.sqrt(np.mean(res)))
    print("MAE =", np.mean(np.abs(pinn_prices - anal_prices)))

    if return_values:
        return spot_grid, tau_grid, res, pinn_prices, anal_prices
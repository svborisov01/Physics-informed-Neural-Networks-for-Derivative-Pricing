import numpy as np
from support_tools.analytical_pricing_tools import heston_call_price, heston_put_price

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
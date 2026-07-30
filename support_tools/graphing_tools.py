import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def plot_convergence(histories, x_axis: str = "epochs"):
    """Plot training loss convergence (log scale) vs epoch or elapsed time."""
    x_axis = x_axis.replace("time_elapsed", "elapsed_time")

    plt.figure(figsize=(10, 6), dpi=200)
    x_label = "Epoch" if x_axis == "epochs" else "Time elapsed (seconds)"

    if isinstance(histories, dict):
        x_vals = (
            histories["epoch"]
            if x_axis == "epochs"
            else histories["elapsed_time"]
        )
        plt.plot(
            x_vals,
            np.log(histories["total"]),
            label=(
                f"{histories['model depth']} layers, "
                f"{histories['model width']} neurons each"
            ),
        )
    else:
        for history in histories:
            x_vals = (
                history["epoch"]
                if x_axis == "epochs"
                else history["elapsed_time"]
            )
            plt.plot(
                x_vals,
                np.log(history["total"]),
                label=(
                    f"{history['model depth']} layers, "
                    f"{history['model width']} neurons each"
                ),
            )

    plt.xlabel(x_label)
    plt.ylabel("Log of Total Loss")
    plt.title("Convergence of model training")
    plt.legend()
    plt.show()


def plot_3d_result(
    inputs,
    values: str = "diff",
    title: str = "Interactive PDE solution",
):
    """
    Plot a 3D surface from slice-test results.

    Parameters
    ----------
    inputs : tuple
        (spot_grid, tau_grid, mse_grid, pinn_prices, anal_prices)
    values : {"diff", "pinn", "anal"}
        Which surface to display.
    title : str
        Plot title (defaults to generic label; pass model-specific title
        via visualize_slice_test in model_wrapper.py).
    """
    X, Y = np.meshgrid(inputs[0], inputs[1])
    if values == "diff":
        Z = inputs[2]
        z_title = "Difference between analytical and PINN solutions"
    elif values == "pinn":
        Z = inputs[3]
        z_title = "PINN-based solution"
    else:
        Z = inputs[4]
        z_title = "Analytical solution"

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z.T,
                colorscale="jet",
                opacity=0.8,
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Spot price (S)",
            yaxis_title="Time to maturity (tau)",
            zaxis_title=z_title,
        ),
        autosize=True,
        width=900,
        height=700,
    )

    fig.show()

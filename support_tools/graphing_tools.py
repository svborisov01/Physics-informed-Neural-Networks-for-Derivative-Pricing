from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None


def save_surface_3d(
    S_grid,
    tau_grid,
    Z,
    title: str,
    path,
    z_label: str = "value",
    cmap: str = "viridis",
    elev: float = 28.0,
    azim: float = -55.0,
):
    """
    Save a static matplotlib 3D surface with axes (S, tau, Z).

    Parameters
    ----------
    S_grid, tau_grid : 1d arrays
    Z : 2d array shaped (len(S_grid), len(tau_grid))
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    S_grid = np.asarray(S_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    Z = np.asarray(Z, dtype=float)
    Tau, SS = np.meshgrid(tau_grid, S_grid)

    fig = plt.figure(figsize=(9.2, 6.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        SS,
        Tau,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.95,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Spot price S")
    ax.set_ylabel("Time to maturity τ")
    ax.set_zlabel(z_label)
    ax.set_title(title, fontsize=11, pad=10)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=z_label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_surface_3d_html(
    S_grid,
    tau_grid,
    Z,
    title: str,
    path,
    z_label: str = "value",
    colorscale: str = "Viridis",
):
    """Save an interactive Plotly 3D surface (S, tau, Z) as HTML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    S_grid = np.asarray(S_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    Z = np.asarray(Z, dtype=float)
    # Plotly Surface: x,y are 2d or 1d; z shape matches
    X, Y = np.meshgrid(S_grid, tau_grid)  # rows=tau, cols=S
    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z.T,
                colorscale=colorscale,
                colorbar=dict(title=z_label),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Spot price S",
            yaxis_title="Time to maturity τ",
            zaxis_title=z_label,
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


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
        # Prefer absolute price error for visualization. inputs[2] is an MSE grid
        # (squared residuals) used for metrics; plotting it exaggerates ATM ridges.
        if len(inputs) >= 5:
            Z = np.abs(np.asarray(inputs[3]) - np.asarray(inputs[4]))
        else:
            Z = np.sqrt(np.maximum(np.asarray(inputs[2]), 0.0))
        z_title = "Absolute difference between analytical and PINN solutions"
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

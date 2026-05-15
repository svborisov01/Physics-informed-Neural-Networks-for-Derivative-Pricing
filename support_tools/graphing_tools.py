import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_convergence(histories, x_axis: str = "epochs"):
    plt.figure(figsize = (10,6), dpi = 200)
    x_label = 'Epoch' if x_axis == 'epochs' else "Time elapsed (seconds)"

    if isinstance(histories, dict):
        x_vals = histories['epoch'] if x_axis == 'epochs' else histories['elapsed_time']
        plt.plot(x_vals, np.log(histories["total"]), label = f'{histories['model depth']} layers, {histories['model width']} neurons each')
    else:
        for history in histories:
            x_vals = history['epoch'] if x_axis == 'epochs' else history['elapsed_time']
            plt.plot(x_vals, np.log(history["total"]), label = f'{history['model depth']} layers, {history['model width']} neurons each')

    plt.xlabel(x_label)
        
    plt.ylabel("Log of Total Loss")
    plt.title("Convergence of model training")
    plt.legend()
    plt.show()

def plot_3d_result(inputs, values: str = "diff"):

    X, Y = np.meshgrid(inputs[0], inputs[1])
    if values == 'diff':
        Z = inputs[2]
        title = 'Difference between Quasi-Analytical and PINN-Based solutions'
    elif values == 'pinn':
        Z = inputs[3]
        title = 'PINN-Based solution'
    else:
        Z = inputs[4]
        title = 'Quasi-Analytical solution'

    fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=Z.T,
            colorscale='jet',
            opacity=0.8,
        )])

    fig.update_layout(
            title='Interactive Black-Scholes PDE solution',
            scene=dict(
                xaxis_title='Spot price (S)',
                yaxis_title='Time (tau)',
                zaxis_title=title,
            ),
            autosize=True,
            width=900,
            height=700,
        )

    fig.show()
    
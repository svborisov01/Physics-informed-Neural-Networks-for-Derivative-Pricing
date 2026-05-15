import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_convergence(histories, x_axis: str = "epochs"):
    plt.figure(figsize = (10,6), dpi = 200)

    if isinstance(histories, dict):
        if x_axis == 'epochs':
            x_axis = histories['epoch']
        else:
            x_axis = histories['elapsed_time']
        plt.plot(x_axis, np.log(histories["total"]), label = f'{histories['model depth']} layers, {histories['model width']} neurons each')
    else:
        for history in histories:   
            if x_axis == 'epochs':
                x_axis = history['epoch']
            else:
                x_axis = history['elapsed_time']
            plt.plot(x_axis, np.log(history["total"]), label = f'{history['model depth']} layers, {history['model width']} neurons each')

    if x_axis == 'epochs':
        plt.xlabel("Epoch")
    else:
        plt.xlabel('Time elapsed (seconds)')
        
    plt.ylabel("Log of Total Loss")
    plt.title("Convergence of model training")
    plt.legend()
    plt.show()
    
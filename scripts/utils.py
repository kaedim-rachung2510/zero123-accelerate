import matplotlib.pyplot as plt
import numpy as np

def plot_subplots(images, titles=[], fig_size=10):
    n_cols = 4
    n_rows = int(np.ceil(len(images) / n_cols))
    plt.figure(figsize=(fig_size, fig_size//3*n_rows))
    for i,img in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1, xticks=[], yticks=[])
        if len(titles):
            plt.title(titles[i][0], color=titles[i][1])
        else:
            plt.title(i)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
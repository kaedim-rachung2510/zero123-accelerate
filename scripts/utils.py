import matplotlib.pyplot as plt
import numpy as np

def plot_subplots(images, titles=[], n_cols=4, fig_size=0):
    if fig_size == 0:
        fig_size = 2.5 * n_cols
    n_rows = int(np.ceil(len(images) / n_cols))
    plt.figure(figsize=(fig_size, fig_size//3*n_rows))
    for i,img in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1, xticks=[], yticks=[])
        if len(titles):
            title = titles[i]
            _title = title if type(title) == str else title[0]
            _color = "k" if type(title) == str else title[1]
            plt.title(_title, color=_color)
        else:
            plt.title(i)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
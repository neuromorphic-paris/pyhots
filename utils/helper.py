import matplotlib.pyplot as plt
import numpy as np

def plot_centers(centers, activations):
    plt.close('all')
    edge_length = np.sqrt(len(centers)).astype(int)
    fig, axes = plt.subplots(edge_length, edge_length)
    axes = np.reshape(axes, -1)

    for index, axis in enumerate(axes):
        axisImag = axis.imshow(centers[index], vmin=0, vmax=1, cmap = plt.cm.hot, interpolation = 'none', origin = 'upper')
        axis.set_ylabel(str(int(activations[index])), rotation=0, labelpad=25)
        axis.set_xticks([])
        axis.set_xticklabels('')
        axis.set_yticks([])
        axis.set_yticklabels('')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axisImag, cax=cbar_ax)
    text = fig.suptitle('Cluster centers')
    
def create_histograms(labels, n_of_centers):
    hists = []
    [hists.append(np.histogram(x, bins=np.arange(0,n_of_centers+1))[0]/len(x)) for x in labels]
    return hists
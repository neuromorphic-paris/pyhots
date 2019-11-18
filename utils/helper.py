import matplotlib.pyplot as plt
import numpy as np

def plot_centers(centers, activations):
    plt.close('all')
    edge_length = np.sqrt(len(centers)).astype(int)
    fig, axes = plt.subplots(edge_length, edge_length)
    axes = np.reshape(axes, -1)
    total_activation = np.sum(activations)
    total_clusters = len(centers)
    for index, axis in enumerate(axes):
        axisImag = axis.imshow(centers[index], vmin=0, vmax=1, cmap = plt.cm.hot, interpolation = 'none', origin = 'upper')
        axis.set_ylabel(str(round(activations[index]/total_activation*total_clusters, 1)), rotation=0, labelpad=17)
        axis.set_xticks([])
        axis.set_xticklabels('')
        axis.set_yticks([])
        axis.set_yticklabels('')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axisImag, cax=cbar_ax)
    text = fig.suptitle('Cluster centers')
    
def create_histograms(labels, n_of_centers):
    return [np.histogram(x, bins=np.arange(0, n_of_centers+1))[0] for x in labels]

def mask_isolated(events, filter_time, ordering, sensor_size):
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    events_copy = np.zeros(events.shape, dtype=events.dtype)
    copy_index = 0
    width = int(sensor_size[0])
    height = int(sensor_size[1])
    timestamp_memory = np.zeros((width, height), dtype=events.dtype)
    for event in events:
        x = int(event[x_index])
        y = int(event[y_index])
        t = event[t_index]
        timestamp_memory[x, y] = t + filter_time
        if ((x > 0 and timestamp_memory[x-1, y] > t) or (x < width-1 and timestamp_memory[x+1, y] > t)
             or (y > 0 and timestamp_memory[x, y-1] > t) or (y < height-1 and timestamp_memory[x, y+1] > t)):
            events_copy[copy_index] = event
            copy_index += 1
    return events_copy[:copy_index]

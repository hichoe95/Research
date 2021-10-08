import torch
import torch.nn as nn
import torchvision
import numpy as np
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import gridspec 


def plus_activations(features):

    mask = features > 0.
    mean = np.mean(mask, axis = 0)
    p_index = np.where(mean == 1.)

    return p_index

    
def sets(features, index, sample_num, range_):

	sets_same = np.arange(range_)
	sets_diff = np.arange(raneg_)

	features_b = featuers > 0

	for i in range(sample_num):

		sign = features[i, index]

		same_sign = np.where(features_b[i] == sign)
		diff_sign = np.where(features_b[i] == ~sign)

		sets_same = np.intersect1d(sets_same, same_sign)
		sets_diff = np.intersect1d(sets_diff, same_sign)


	return sets_same, sets_diff


def print_channelwise(sets, feature_shape, height, width):

    channels = np.unique(sets) // (feature_shape[1] ** 2)

    zeros = np.zeros(feature_shape[0] * feature_shape[1] * feature_shape[2])
    zeros[sets] = 1
    zeros = zeros.reshape(feature_shape)

    gs = gridspec.GridSpec(height, width, wspace = 0.0, hspace = 0.2)

    plt.tight_layout()
    plt.figure(figsize = (width*2, height*2))

    for i, c in enumerate(channels[:height*width]):
        plt.subplot(gs[i//width, i%width])
        plt.axis('off')
        plt.title('{}'.format(c))
        plt.imshow(zeros[c], vmin = -1, vmax = 1, cmap = 'RdBu_r')
    plt.show()


def coord(index, size):
    channel = index // (size * size)
    
    n = index % (size * size)
    
    x = n // size
    y = n % size
    
    return channel, x, y



    
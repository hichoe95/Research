import torch
import torch.nn as nn
import torchvision
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib import gridspec 


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
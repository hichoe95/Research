import torch
import numpy as np

from extract_feature import feature_extractor


def print_genboundary(model, x_range, y_range, dir1, dir2, latent_w, arange, resolution = 30):
	with torch.no_grad():
		x = np.linspace(x_range, resolution)
		y = np.linsapce(y_range, resolution)
		xx, yy = np.meshgrid(x, y)

		features_ptb = []

		for a, b in zip(xx, yy):
			for aa, bb in zip(a, b):
				pca_d = (dir1 * aa + dir2 * bb)
				w = go_direction(torch.tensor(latent_w).unsqueeze(1).repeat(1,18,1), arange, pca_d)
				features_ptb.append(feature_extractor(model, 'synthesis.layer4', w, synthesis_layer = True).flatten())

		features_ptb = np.array(features_ptb)

	for i in range(0, 50):
		plt.contour(xx, yy, features_ptb[:,i].reshape(resolution, resolution), levels = 0)

	plt.show()
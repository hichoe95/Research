import torch
import torch.nn as nn
import numpy as np

from directions import *
from extract_feature import feature_extractor

class boundary():
	def __init__(self, model, layer, x_range, y_range, pc1, pc2, latent_w, arange, alpha = 1, resolution = 30):

		assert latent_w.shape == (1, 512), 'latent_w.shape == (1,512), but got {}'.format(latent_w.shape)

		self.model = model
		self.layer = layer
		self.x_range = x_range
		self.y_range = y_range
		self.pc1 = pc1
		self.pc2 = pc2
		self.latent_w = latent_w
		self.arange = arange
		self.resolution = resolution

		self.alpha = 1

	def feature_grid(self, synthesis_layer = True, w18 = None):

		with torch.no_grad():

			xs, xe = self.x_range
			ys, ye = self.y_range 

			x = np.linspace(xs, xe, self.resolution)
			y = np.linspace(ys, ye, self.resolution)

			xx, yy = np.meshgrid(x, y)

			if w18 is not None:
				new_pc = np.zeros((1, 18, 512))
				new_pc[:, self.arange] = self.pc1

				origin = np.repeat(np.expand_dims(self.latent_w, axis = 1), 18, axis = 1)

				x_ptb = xx.flatten().reshape(-1, 1, 1) * new_pc
				y_ptb = yy.flatten().reshape(-1, 1, 1) * ((origin - w18)/np.linalg.norm(origin-w18) * self.alpha)

				grid_ptb = (x_ptb + y_ptb).reshape(-1, 18, 512)

			else:
				x_ptb = xx.flatten().reshape(-1, 1) * self.pc1
				y_ptb = yy.flatten().reshape(-1, 1) * self.pc2

				grid_ptb = (x_ptb + y_ptb).reshape(-1, 1, 512)

			f = feature_extractor(self.model, self.layer, torch.tensor(self.latent_w).unsqueeze(1).repeat(1, 18, 1).to(device), synthesis_layer = True)
			f_shape = f.shape
			f = f.flatten()
			index = (-f).argsort()

			features_ptb = []



			for i in range(0, self.resolution):
				pca_d = grid_ptb[i * self.resolution : (i+1) * self.resolution]
				if w18 is not None:
					w = (torch.tensor(self.latent_w, dtype = torch.float32).unsqueeze(1).repeat(self.resolution, 18, 1) + torch.tensor(pca_d, dtype = torch.float32)).to(device)
				else:
					if synthesis_layer:
						w = go_direction(torch.tensor(self.latent_w).unsqueeze(1).repeat(self.resolution, 18, 1).to(device), self.arange, pca_d)
						features_ptb.append(feature_extractor(self.model, self.layer, w, synthesis_layer = True).reshape(self.resolution, -1))

					else:
						z = go_direction(torch.tensor(self.latent_w).unsqueeze(1).repeat(self.resolution,1, 1).to(device), self.arange, pca_d, synthesis_layer = False)
						features_ptb.append(feature_extractor(self.model, self.layer, z.squeeze(), synthesis_layer = False).reshape(self.resolution, -1))


			features_ptb = np.array(features_ptb).reshape(self.resolution ** 2, -1)

		return xx, yy, index, features_ptb, f, f_shape

	def print_boundary(self, xx, yy, features, index, res = 40, topn = 300, figsize = (10,10)):

		plt.figure(figsize = figsize)


		for i in range(0, topn):
			plt.contour(xx, yy, features[:, index[i]].reshape(res, res), levels = [0 + 1e-7, 0 - 1e-7], alpha = 0.4)

		plt.show()

	def print_boundary_images(self, res = 20, w18 = None):

		xs, xe = self.x_range
		ys, ye = self.y_range

		x = np.linspace(xs, xe, res)
		y = np.linspace(ys, ye, res)

		xx, yy = np.meshgrid(x, y)

		if w18 is not None:
			new_pc = np.zeros((1, 18, 512))
			new_pc[:, self.arange] = self.pc1

			origin = np.repeat(np.expand_dims(self.latent_w, axis = 1), 18, axis = 1)

			x_ptb = xx.flatten().reshape(-1, 1, 1) * new_pc
			y_ptb = yy.flatten().reshape(-1, 1, 1) * ((origin - w18)/np.linalg.norm(origin-w18) * self.alpha)

			grid_ptb = (x_ptb + y_ptb).reshape(-1, 18, 512)

		else:
			x_ptb = xx.flatten().reshape(-1, 1) * self.pc1
			y_ptb = yy.flatten().reshape(-1, 1) * self.pc2

			grid_ptb = (x_ptb + y_ptb).reshape(-1, 1, 512)

		images_ptb = []

		with torch.no_grad():
			for i in tqdm(range(0, res)):
				pca_d = grid_ptb[i * res : (i+1) * res]
				if w18 is not None:
					w = (torch.tensor(self.latent_w, dtype = torch.float32).unsqueeze(1).repeat(res, 18, 1) + torch.tensor(pca_d, dtype = torch.float32)).to(device)
				else:
					w = go_direction(torch.tensor(self.latent_w).unsqueeze(1).repeat(res, 18, 1).to(device), self.arange, pca_d)
				images_ptb.append(self.model.synthesis(w)['image'].detach().cpu().numpy())

		images_ptb = np.array(images_ptb)

		gs = gridspec.GridSpec(res, res, wspace = 0.05, hspace = 0.02)
		plt.figure(figsize = (res, res))

		for k in range(res):
			for l in range(res):
				plt.subplot(gs[res - 1 - k, l%res])
				plt.axis('off')
				plt.title('{}'.format(l))
				plt.imshow(minmax(images_ptb[k][l].transpose(1,2,0)))

		plt.show()

	def sorted_index(self, cor1, cor2, w18 = None):

		x1, y1 = cor1
		x2, y2 = cor2

		if w18 is not None:
			range_ = np.arange(0, 18)
			pc1 = np.zeros((1, 18, 512))
			pc1[:, self.arange] = self.pc1

			ori = np.repeat(np.expand_dims(self.latent_w, axis = 1), 18, axis = 1)
			pc2 = ((ori - w18)/np.linalg.norm(ori - w18) * self.alpha)

		else:
			range_ = self.arange
			pc1 = self.pc1
			pc2 = self.pc2

		with torch.no_grad():
			w0 = go_direction(torch.tensor(self.latent_w).unsqueeze(1).repeat(1, 18, 1).to(device), range_, x1 * pc1 + y1 * pc2)
			w1 = go_direction(torch.tensor(self.latent_w).unsqueeze(1).repeat(1, 18, 1).to(device), range_, x2 * pc1 + y2 * pc2)

			feature = feature_extractor(self.model, self.layer, torch.cat([w0, w1], dim = 0), synthesis_layer = True)
			feature = feature.reshape(2, -1)

			index_ = np.where((feature[0]>= 0) != (feature[1]>=0))[0]
			index_sorted = sorted(list(index_), key = lambda x : -abs(feature[0][x]) - abs(feature[1][x]))

		return w0, w1, index_sorted, feature

#일단은 인덱스에 해당하는 뉴런들을 표시하고 채널 방향으로 싹 다 더해서 mask로 사용하고 있음.

def boundary_mask(feature, index, mask_shape, out_shape):
    assert len(feature.shape) == 2, 'feature should be flattened!'
    
    mask = -torch.ones(feature.shape[-1])
    mask[index] = 1
    
    mask = mask.view(mask_shape)
    mask_sum = mask.sum(dim = 0)
    
    assert out_shape[-1] % mask_shape[-1] == 0, 'out_shape[0] and out_shape[1] should be devided by mask_shape[-1].'
    
    m = nn.Upsample(scale_factor = out_shape[-1]//mask_shape[-1], mode = 'bilinear')
    out_mask = m(mask_sum.view(1,1, mask_shape[-1], mask_shape[-1]))
    
    return out_mask.view(out_shape)
    
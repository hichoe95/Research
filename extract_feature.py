import torch
import torch.nn as nn
import torchvision
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib import gridspec 



device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

layers = ['layer2', 'layer4', 'layer6', 'layer8', 'layer10', 'layer12', 'layer14', 'layer16']



def feature_extractor(model, layer, input, synthesis_layer = False):
	feature_map = []

	def fn(m, i, o):
		feature_map.append(o[0].detach().cpu().numpy())
		hook.remove()

	hook = eval('model.'+layer+'.register_forward_hook(fn)')
	with torch.no_grad():
		try:
			if synthesis_layer:
				model.synthesis(input)
			else:
				model(input)
		except Exception as e:
			hook.remove()
			print(e)
			print(input.shape)
			print('You should check the inputs.')


	return feature_map[0].squeeze()

# only styleGANv2
def all_features(model, latent, layers = layers):
	return {layer : feature_extractor(model, 'synthesis.'+layer, latent) for layer in layers}


def feature_change(model : nn.Module, layer : str, mask : torch.tensor, index : list, w : np.array):

	def modify_feature(mask, index):
		def fn(m, i, o):
			for i in index:
				o[0][0][i] = o[0][0][i].masked_fill(mask = mask.to(device), value = o[0][0][i].max())
			hook.remove()
			return o
		return fn

	hook = eval('model.' + layer + '.register_forward_hook(modify_feature(mask, index))')

	assert w.shape == (1,512), 'w vector should have dimension of batch_size(1)'

	with torch.no_grad():
		try:
			modified_image = model.synthesis(torch.tensor(w).unsqueeze(1).repeat(1,18,1).to(device))['image']
		except:
			hook.remove()

	return modified_image



# all feature plot 
def print_features(features, width = 16):

	num_features = features.shape[0]

	height = num_features//width

	assert width * height == num_features, 'width * (features//width) == features.'

	gs = gridspec.GridSpec(height, width, wspace = 0.0, hspace = 0.1)

	plt.figure(figsize = (width, height))
	plt.tight_layout()

	for i in range(height):
		for j in range(width):
			plt.subplot(gs[i,j])
			plt.axis('off')
			plt.title(f'{i*width + j}')
			plt.imshow(features[i * width + j]) 
	plt.show()




import torch
import torchvision
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib import gridspec 



device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

layers = ['layer2', 'layer4', 'layer6', 'layer8', 'layer10', 'layer12', 'layer14', 'layer16']


### w 벡터 이동시키는 것 까지 만들기 ! 

def feature_extractor(model, layer, input, synthesis_layer = False):
	feature_map = []

	def fn(m, i, o):
		feature_map.append(o[0].detach().cpu().numpy())
		hook.remove()

	hook = eval('model.'+layer+'.register_forward_hook(fn)')

	try:
		if synthesis_layer:
			model.synthesis(input)
		else:
			model(input)
	except:
		hook.remove()
		print('You should check the inputs.')

	# print(np.array(feature_map).shape)

	return feature_map[0].squeeze()

# only styleGANv2
def all_features(model, latent):
	return {layer : feature_extractor(model, 'synthesis.'+layer, latent) for layer in layers}



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




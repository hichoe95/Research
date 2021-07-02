from PIL import Image

import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

from matplotlib import gridspec
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms
import cv2

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

def image_resize(image, size = (512,512)):
	image = torch.tensor(image, dtype = torch.float32)
	image = ToPILImage()(image)
	image_resize = image.resize(size)

	image_resize = ToTensor()(image_resize)

	return image_resize[0] != 0

def minmax(image):
	return (image - image.min())/(image.max() - image.min())



def IOU(mask, input, thres = 0.):
	input_mask = input > thres

	IOU = ((mask.int() + input_mask.int()) == 2).sum()
	IOU = 1. * IOU/(mask.size(1)**2)

	return IOU


def all_IOU(mask, inputs, thres = 0.):
	return np.array([IOU(mask, torch.tensor(inputs[i]), thres = thres) for i in range(inputs.shape[0])])


def latents(model = None, num = 1000000, type = 'ffhq'):

	if type == 'ffhq':
		l = np.load('../GAN_analy/genforce/latent_z.npy')
		l2 = np.load('../GAN_analy/genforce/latent_z_2.npy')
		latent_z = np.concatenate([l, l2], axis = 0)

		w = np.load('../GAN_analy/genforce/latent_w.npy')
		w2 = np.load('../GAN_analy/genforce/latent_w_2.npy')
		latent_w = np.concatenate([w,w2.reshape(-1,512)], axis = 0)
	else:
		z = torch.randn(num, 512).to(device)
		w = torch.zeros(num, 512)

		with torch.no_grad():
			for i in range(num // 10000):
				w[i * 10000 : (i+1) * 10000] = model.mapping(z[i * 10000 : (i+1) * 10000])['w'].detach().cpu()

		latent_z = z.detach().cpu().numpy()
		latent_w = w.numpy()

	return latent_z, latent_w


def img_process(img_path):
	
	to_tensor = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	    ])

	img = Image.open(img_path)
	image = img.resize((512,512), Image.BILINEAR)
	img = to_tensor(image)
	img = torch.unsqueeze(img, 0)

	return img

def print_two_images(image1, image2):
	gs = gridspec.GridSpec(1, 2, wspace = 0.1, hspace = 0.1)

	plt.figure(figsize = (10, 5))
	plt.tight_layout()

	plt.subplot(gs[0,0])
	plt.imshow(minmax(image1[0].detach().cpu().permute(1,2,0)))

	plt.subplot(gs[0,1])
	plt.imshow(minmax(image2[0].detach().cpu().permute(1,2,0)))

	plt.show()




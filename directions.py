from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy

from estimator import *
from misc import *

torch.manual_seed(45)
np.random.seed(45)

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

# latent_z, latent_w = latents()


def projection(pc1, pc2, latent):
    x = np.dot(pc1, latent)
    y = np.dot(pc2, latent)

    return x, y

def GANSpace_dir(model, estimator = 'ipca', z_nums = 1e6, components = 80, alpha = 1):
    z = torch.randn(int(z_nums), 512).to(device)    
    w = np.ones((int(z_nums), 512), dtype = np.float32)

    with torch.no_grad():
        for i in range(int(z_nums)//10000):
            w[i*10000 : (i+1) * 10000] = model.mapping(z[i*10000 : (i+1) * 10000])['w'].detach().cpu().numpy()


        del model
        torch.cuda.empty_cache()


    w_global_mean = w.mean(axis = 0, keepdims = True, dtype = np.float32)
    w -= w_global_mean

    transformer = get_estimator(estimator, components, alpha = alpha)
    transformer.fit(w)

    w_comp, w_stdev, w_var_ratio = transformer.get_components()
    w_comp /= np.linalg.norm(w_comp, axis = -1, keepdims = True)

    return w_comp, w_stdev, w_global_mean


def pca_direction(range_, pca_num, indice, weight=None):
    vectors_ = latent_w[indice[:range_]]    
    
    # normalizing vectors
    vectors = vectors_ - vectors_.mean(axis = 0)
    vectors = vectors/vectors_.std(axis=0)

    u, s, vt = np.linalg.svd(vectors, full_matrices = True)

    if weight is None:
        weight = 1-s[-pca_num:]/s[-pca_num:].sum()
    
    return vt[-pca_num:].T@weight * vectors_.std(axis=0), s, vectors_


def go_direction(ws, layers, direction, use_norm = False):
    w = copy.deepcopy(ws.detach())

    w[:,layers] += torch.tensor(direction, dtype = torch.float32).to(device)
    
    # norm_ = torch.norm(w[0, layers[0]])
    
    return w

def print_image_movement(Gs_style, latent_z, latent_w, alpha, range_, pca_d, use_norm = False, type = 'ffhq'):
    with torch.no_grad():
        if type == 'ffhq':
            w = torch.tensor([latent_w]).unsqueeze(1).repeat(1,18,1).to(device)
        elif type == 'car':
            w = torch.tensor([latent_w]).unsqueeze(1).repeat(1,16,1).to(device)
        
        gs = gridspec.GridSpec(1, 3, wspace = 0.01, hspace = 0.1)
        plt.figure(figsize = (14, 7))
        plt.tight_layout()

        # changed image
        plt.subplot(gs[0,1])
        w_ = go_direction(w, range_, -alpha * pca_d.reshape(1,1,-1), use_norm)
        changed_image = Gs_style.synthesis(w_)
        plt.imshow(minmax(changed_image['image'][0].detach().cpu().numpy().transpose(1,2,0)))
        plt.title('After(-)')
        plt.axis('off')

        # original image
        plt.subplot(gs[0,0])
        plt.axis('off')
        sample = Gs_style(torch.tensor([latent_z]).to(device))
        plt.title('Before')
        plt.imshow(minmax(sample['image'][0].detach().cpu().numpy().transpose(1,2,0)))

        plt.subplot(gs[0,2])
        w_ = go_direction(w, range_, alpha * pca_d.reshape(1,1,-1), use_norm)
        changed_image = Gs_style.synthesis(w_)
        plt.imshow(minmax(changed_image['image'][0].detach().cpu().numpy().transpose(1,2,0)))
        plt.title('After(+)')
        plt.axis('off')

        plt.show()


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data




def print_image(image, show = False):
    image = image.detach().cpu().numpy()
    
    image = adjust_dynamic_range(image, [image.min(), image.max()], [0,1])
    
    plt.imshow(image[0].transpose(1,2,0))
    
    if show:
        plt.show()
    
def d_outputs(module : torch.nn.Module, name):
    features = []
   
    def fn(_, __, out):
        features.append(out.detach().cpu().numpy())
        
    hook = eval('module.'+name+'.register_forward_hook(fn)')
    
    for i in tqdm(range(0,30000)):
        #1번 부터 시작.
        image_arr = [np.array(Image.open(f'../GANCP/celebA_analysis/data1024x1024/{i+1:05d}.jpg'))]
        image_arr = np.array(image_arr).transpose(0,3,1,2)
        
        input_batch = adjust_dynamic_range(image_arr, [0, 255], [0,1])
        
        with torch.no_grad():
            module(torch.Tensor(input_batch).to(device))
    
    hook.remove()

    return features



class find_data_any():
    def __init__(self, Gs_style, D_style, query, query_index, reshape_size = 30000, real_images = '../GANCP/celebA_analysis/data1024x1024/', real_features = './features_4x4.npy',  output_layer = 'layer16', random = True, Gen = False):
        self.Gs_style = Gs_style
        self.D_style = D_style
        self.Gen = Gen
        
        self.random = random
        self.output_layer = output_layer
        
        self.query = query
        self.query_index = query_index
        self.nidx = 0
        
        self.real_images = real_images
        
        self.reshape_size = reshape_size
        
        self.real_images_feature = np.load(real_features).reshape(reshape_size,-1)
        self.real_images_sign = self.real_images_feature > 0
        self.dimension = self.real_images_sign.shape[-1]
        print("features_4x4.shape : ", self.real_images_sign.shape)
        
        self.index = None
        self.gen_image, self.gen_image_scale = None, None
        self.gen_image_feature = None
        self.gen_image_sign = None
        self.how_many_sign_equal = None
        self.distance = None
        
        self.get_index()
        self.generate_image()
    
    
    def get_index(self,):
        if self.random:
            self.index = np.random.choice(self.query_index)
        else:
            self.index = self.query_index[self.nidx]
            self.nidx += 1
            if self.nidx >= len(self.query_index):
                self.nidx = 0
                print('Index is out of range')
                return
    
    def custom(self, image):
        self.gen_image_scale = minmax(image).transpose(2,0,1).reshape(1,3,1024,1024)
        self.gen_image_feature = self.image_feature(self.gen_image_scale).reshape(1,-1)
        self.gen_image_sign = self.gen_image_feature >= 0
#         print(self.gen_image_sign.shape)
        self.how_many_sign_equal = self.is_sign_equal()
        self.distance = self.cal_distance()
    
    def generate_image(self, ind = -1):
        
        with torch.no_grad():
            if ind == -1:
                self.get_index()
                self.gen_image = self.Gs_style(torch.tensor([self.query[self.index,...]], dtype = torch.float32).to(device))
            else:
                self.gen_image = self.Gs_style(torch.tensor([self.query[ind,...]], dtype = torch.float32).to(device))
            self.gen_image = self.gen_image['image'].detach().cpu().numpy()
            
            print('gen_image.shape : ', self.gen_image.shape)
            
            self.gen_image_scale = self.adjust_dynamic_range(self.gen_image, [self.gen_image.min(), self.gen_image.max()], [0,1])
            self.gen_image_feature = self.image_feature(self.gen_image_scale, ind).reshape(1,-1)
            self.gen_image_sign = self.gen_image_feature >= 0
            self.how_many_sign_equal = self.is_sign_equal()
            self.distance = self.cal_distance()
        
    def adjust_dynamic_range(self, data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def image_feature(self, x, ind = 0):
        with torch.no_grad():
            gen_feature = []

            if not self.Gen:
                def fn(_, __, o):
                    gen_feature.append(o.detach().cpu().numpy())
                hook = eval('self.D_style.'+self.output_layer+'.register_forward_hook(fn)')
                self.D_style(torch.tensor(x).to(device))
            else:
                def fn(_, __, o):
                    gen_feature.append(o[0].detach().cpu().numpy())
                hook = eval('self.Gs_style.'+self.output_layer+'.register_forward_hook(fn)')
                self.Gs_style(torch.tensor([self.query[ind,...]]).to(device))
                
            hook.remove()
        return gen_feature[0]
    
    
    def is_sign_equal(self,):
#         how_many_sign_equal = np.array([(~(self.gen_image_sign ^ self.real_images_sign[index])).sum() for index in range(0,self.reshape_size)])
        how_many_sign_equal = np.array([(self.gen_image_sign == self.real_images_sign[index]).sum() for index in range(0,self.reshape_size)])
        return how_many_sign_equal
    
    def cal_distance(self,):
        distance = np.array([np.linalg.norm(self.gen_image_feature[0,...] - self.real_images_feature[i,...], axis = 0) for i in range(self.reshape_size)])
        return distance
    
    def minmax_scaler(self, x):
        return (x - x.min()) / (x.max() - x.min())
            
    def forward(self, topn, low = False, show = True, dist = False):
        indice = []
        
        indice = np.argsort(self.how_many_sign_equal) if low else np.argsort(-self.how_many_sign_equal)
        
        indice_dist = []
        
        if dist:
            indice_dist = np.argsort(-self.distance) if low else np.argsort(self.distance)

        if show:
            self.plot_figure(indice[:topn], indice_dist[:topn], dist)
        
        # index of image starts from 1 to 30000. 
        return indice
    
    def plot_figure(self, indice, indice_dist, dist = False):
        if dist:
            gs = gridspec.GridSpec(3, indice.shape[0] + 1, wspace = 0.0, hspace = 0.1)
        else:
            gs = gridspec.GridSpec(1, indice.shape[0] + 1, wspace = 0.0, hspace = 0.1)

        plt.figure(figsize = (50,11))
        plt.tight_layout()
        
        plt.subplot(gs[0,0])
        plt.axis('off')
        plt.title('Generated Image', fontsize=20)
        print
        plt.imshow(self.gen_image_scale[0].transpose(1,2,0))
        
        print('indice : ',indice)
        with torch.no_grad():
            reals = self.Gs_style(torch.tensor(self.query[indice]).to(device))['image'].detach().cpu().numpy()
#         reals = np.array([np.array(Image.open(self.real_images+'{:05d}.jpg'.format(i+1))) for i in indice])
        
        for i in range(len(indice)):
            plt.subplot(gs[0,i+1])
            plt.axis('off')
            plt.title(f'{self.how_many_sign_equal[indice[i]]:d} / {self.dimension:d}',fontsize = 20)
            plt.imshow(self.minmax_scaler(reals[i].transpose(1,2,0)))
        
        if dist:
            reals = np.array([np.array(Image.open(self.real_images+'{:05d}.jpg'.format(i+1))) for i in indice_dist])
            for i in range(len(indice)):
                plt.subplot(gs[1,i+1])
                plt.axis('off')
                plt.title(f'{self.distance[indice_dist[i]]:.3f}', fontsize = 20)
                plt.imshow(reals[i])

        plt.show()
        
    def __call__(self, topn, low = False, show = True, dist = False):
        return self.forward(topn = topn, low = low, show = show, dist = dist)
    
    

def minmax(x):
    
    if x.max() == x.min():
        return (x-x.min())/(x.max() - x.min() + 1e-7)
    return (x - x.min())/(x.max() - x.min())



def dist_neighbor(feature_layer):
    # feature_layer.shape = (training_iter_num, batch_size, channel, height, width)
    dist = []
    C = feature_layer[0].shape[1]
    for c in range(C):
        dist_each = 0.
        for t in range(1,2000):
            a = minmax(feature_layer[t][0][c])
            b = minmax(feature_layer[t-1][0][c])
            dist_each += np.abs((a - b).mean())

        dist.append(dist_each)
    return dist



def show_features(feature_layer: list, channel_index_list: list, width: int , threshold = 0. , show_index = False, use_intb = True):

    H = channel_index_list
    W = width

    gs = gridspec.GridSpec(len(H), W, wspace = 0.0, hspace = 0.1)

    plt.figure(figsize = (W*2, len(H)*2))
    plt.tight_layout()
    intb = 2000//W if use_intb else 1
    
    for i,h in enumerate(H):
       # plt.title('i')
        for w in range(W):
            plt.subplot(gs[i,w])
            plt.axis('off')
            plt.title(f'{intb*w}')
            plt.imshow(feature_layer[intb*w][0][h])

    plt.show()
    
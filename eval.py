import os
import numpy as np
import time
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import face_alignment

from dataset.confer_dataset import *
from dataset.vox_lips_dataset import *
from dataset.prepare_data import *
from nets.solver import *
from nets.networks import *
from nets.utils import *


def load_config(save_dir):
    
    config = Config()
    
    # Args
    args_path = os.path.join(save_dir, 'args')
    if os.path.isfile(args_path):
        with open(args_path, 'rb') as f:
            args = json.load(f)
    else:
        print('Besoin d un fichier arguments !')

    for attr, attr_value in args.items():
        setattr(config, attr, attr_value)
    
    data_dim = 68 * config.data_dim
    setattr(config, 'input_size', data_dim)

    return config


class Model:

    def __init__(self, model_name, model_dir, iteration=None):
        
        for path in os.walk(model_dir):
            if os.path.basename(path[0]) == model_name:
                save_dir = path[0]
                break

        config = load_config(save_dir)
        self.dataset_type = config.dataset

        ## Init model
        self.load(save_dir, config, iteration)
        self.config = config
        self.save_dir = save_dir


    def load(self, save_dir, config, iteration):

        solver = InteractionSolver(config)
        epoch = solver.resume(save_dir, iteration, False)
        print(f'Loading success, epoch: {epoch}')
        solver.eval()
        self.solver = solver


    def test_model(self, inpt, seq_len, rdm_augment, rdm_derivatives):
        
        if rdm_augment:
            aug_seq = torch.Tensor(augment_batch(inpt.numpy()))
            inpt = torch.stack([inpt, aug_seq], dim=1).flatten(end_dim=1)
        else:
            inpt = inpt.repeat_interleave(2, dim=0)

        size, channels, dim = inpt.shape
        if rdm_derivatives:
            t_range = 0.01
            beta = 0.9
            v_0 = np.random.uniform(low=-t_range, high=t_range, size=(size, 2))
            a_0 = (beta - 1) * v_0 + (1 - beta) * np.random.uniform(low=-t_range, high=t_range, size=(size, 2))
            v_0 = torch.Tensor(v_0).unsqueeze(1).repeat(1, channels, 1)
            a_0 = torch.Tensor(a_0).unsqueeze(1).repeat(1, channels, 1)
        else:
            v_0 = torch.zeros_like(inpt)
            a_0 = torch.zeros_like(inpt)
        inpt = torch.cat([inpt, v_0, a_0], dim=1).view(size, 1, -1).cuda()
        with torch.no_grad():
            _, generation = self.solver.dynamical_model(inpt, [2] * int(size / 2), seq_len)
            
        return generation

    def process_dir(self, dir_path, out_dir, seq_len=120, rdm_augment=True, rdm_derivatives=True):

        t = time.time()
        batch, scales, filenames = self.read_dir(dir_path)
        print(f'done extracting landmarks in {time.time() - t} \n generating sequences...')

        sequence = self.test_model(batch, seq_len, rdm_augment, rdm_derivatives)[::2]
        sequence = (sequence.cpu().numpy() * scales[:, np.newaxis, np.newaxis])
        bs, length, _ = sequence.shape
        sequence = sequence.reshape(bs, length, -1, 2)[..., :68, :]
        sequence = resize(sequence)
        print(f'done generating sequences in {time.time() - t} \n saving videos...')

        for i in range(len(sequence)):
            print(i)
            vis = Vis()
            vis.plot_gif(os.path.join(out_dir, f'moving_{filenames[i]}.gif'), sequence[i])
            print(f'vid {i} done in {time.time() - t}')


    def read_dir(self, dir_path):
        
        dataset = []
        scales = []
        filenames = []
        predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)
        for img_path in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path, img_path))
            shapes = predictor.get_landmarks(img)
            if (not shapes or len(shapes) != 1):
                print('Cannot detect face landmarks. Exit.')
                exit(-1)
            shape_2d = shapes[0]
            scale_fact = np.random.uniform(1.05, 1.4)
            scale = shape_2d.max() * scale_fact
            scales.append(scale_fact)
            dataset.append(shape_2d / scale)
            filenames.append(img_path.split('.')[0])
        scales = np.array(scales)
        batch = torch.Tensor(np.stack(dataset))
        return batch, scales, filenames


def augment_batch(sample):
    size, channels, dim = sample.shape
    processed_p = []
    geom_range_y = (sample[..., 1].max(axis=-1) - sample[..., 1].min(axis=-1))
    geom_range_x = (sample[..., 0].max(axis=-1) - sample[..., 0].min(axis=-1))
    max_range = np.vstack([geom_range_y.max(axis=-1), geom_range_x.max(axis=-1)]).max(axis=0)
    mean_range = geom_range_y.mean(axis=-1)

    min_downscales = (0.6 / mean_range)
    rescalings = np.random.uniform(low=min_downscales, high=1.0, size=size)
    flips = np.random.randint(low=0, high=2, size=size)

    for p_idx in np.arange(size):
        p_array = sample[p_idx]

        # Compute parameters of transformation matrix
        origin = 0.5 * (1 - rescalings[p_idx]) * np.ones(2)
        M = affine_matrix(rescalings[p_idx], origin, dim)

        # Rescale and center
        arr = scale_and_translate(p_array, M)

        # Random translation
        o_x = np.random.uniform(low=-arr[..., 0].min(), high=0.99 - arr[..., 0].max())
        o_y = np.random.uniform(low=-arr[..., 1].min(), high=0.99 - arr[..., 1].max())
        M = affine_matrix(1, [o_x, o_y], dim)
        arr = scale_and_translate(arr, M)

        # Maybe flip
        if flips[p_idx] == 1:
            arr[..., 0] = 1 - arr[..., 0]

        processed_p.append(arr)

    sample = np.stack(processed_p)
    return sample


def resize(sequence, ref_size=180):
    sample = (sequence * ref_size).astype(int)
    bs, l, channels, _ = sample.shape
    
    spatial_extents = sample.max(axis=(1, 2)) - sample.min(axis=(1, 2))
    sizes = spatial_extents[:, :2].max(axis=1)
    sizes = (sizes * 1.1).astype(int)
    o_primes = ((sample.min(axis=(1, 2)).astype(int) + sample.max(axis=(1, 2)).astype(int) - sizes.reshape(len(sizes), 1)) / 2).astype(int)
    o_primes = o_primes[:, :2]
    
    processed = []

    for p_idx in range(len(sample)):

        p_array = sample[p_idx]
        S = sizes[p_idx]
        o_prime = o_primes[p_idx]

        # Compute parameters of transformation matrix
        scale_f = ref_size / S
        origin = (-scale_f * o_prime).astype(int)
        M = affine_matrix(scale_f, origin, 2)
        processed.append(scale_and_translate(p_array.reshape(l * channels, 2), M).reshape(l, channels, 2))
        
    return np.stack(processed) / ref_size


class Vis(object):
    def __init__(self):
        
        self.fig = plt.figure()
        self.init_ax()
        
        self.colors = {
            'chin': 'green',
            'eyebrow': 'orange',
            'nose': 'blue',
            'eyes': 'red',
            'outer_lip': 'purple',
            'innner_lip': 'pink'
        }

    def init_ax(self):
        self.ax = self.fig.add_subplot()
        self.ax.cla()
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

    def update(self, f, coord):
        self.ax.invert_yaxis()
        self.ax.scatter(coord[:, 0], coord[:, 1], linewidths=1)

        #chin
        self.ax.plot(coord[0:17,0],coord[0:17,1],marker='',markersize=5,linestyle='-',color=self.colors['chin'],lw=2)
        #left and right eyebrow
        self.ax.plot(coord[17:22,0],coord[17:22,1],marker='',markersize=5,linestyle='-',color=self.colors['eyebrow'],lw=2)
        self.ax.plot(coord[22:27,0],coord[22:27,1],marker='',markersize=5,linestyle='-',color=self.colors['eyebrow'],lw=2)
        #nose
        self.ax.plot(coord[27:31,0],coord[27:31,1],marker='',markersize=5,linestyle='-',color=self.colors['nose'],lw=2)
        self.ax.plot(coord[31:36,0],coord[31:36,1],marker='',markersize=5,linestyle='-',color=self.colors['nose'],lw=2)
        #left and right eye
        self.ax.plot(coord[36:42,0],coord[36:42,1],marker='',markersize=5,linestyle='-',color=self.colors['eyes'],lw=2)
        self.ax.plot(coord[42:48,0],coord[42:48,1],marker='',markersize=5,linestyle='-',color=self.colors['eyes'],lw=2)
        #outer and inner lip
        self.ax.plot(coord[48:60,0],coord[48:60,1],marker='',markersize=5,linestyle='-',color=self.colors['outer_lip'],lw=2)
        self.ax.plot(coord[60:68,0],coord[60:68,1],marker='',markersize=5,linestyle='-',color=self.colors['innner_lip'],lw=2) 


    def plot_mp4(self, save_path, coords, fps=25):
        length = len(coords)
        f = 0

        metadata = dict(title='01', artist='Matplotlib', comment='motion')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        with writer.saving(self.fig, save_path, 100):
            for i in range(length):
                self.init_ax()
                self.update(f, coords[i])
                writer.grab_frame()
                plt.pause(0.01)
                f += 1
        plt.close()

    
    def plot_gif(self, save_path, coords, fps=25):
        length = len(coords)
        interval = 1000 / fps
        
        def update_gif(f):
            self.init_ax()
            self.update(f, coords[f])

        ani = FuncAnimation(self.fig, update_gif, frames=length, interval=interval)
        ani.save(save_path, writer='pillow')
        plt.close()
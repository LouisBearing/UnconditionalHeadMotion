import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import json
from scipy import linalg
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from .pytorch_i3d import *
from .inception import *
from scipy.spatial.transform import Rotation as R
import torch.distributed as dist


class Config:
    
    def __init__(self):
        
        ### General params
        self.save_dir = './models'
        self.save_every = 10
        self.sync_classif_path = './sync_classif'
        self.i3d_path = './inception/rgb_imagenet.pt'
        self.iv3_path = './inception/pt_inception-2015-12-05-6726825d.pth'
        self.velocity = True
        self.velocity_input = True
        self.velocity_output = True
        self.acceleration = True
        self.acceleration_input = False
        self.acceleration_output = False
        self.velocity_ma_range = 3
        self.positions = True
        self.data_dim = 2
        self.obs_len = 25
        self.keypoints = False
        self.jacobian = True
        self.fine_tuning_data = True
        self.range_high = 0.8
        self.range_low = 0.4
        self.optimal_setting = True
        self.mean_s = 0.6
        self.std_s = 0.1
        self.rescaling_uniform = True
        self.n_pers = 2
        self.dataset = 'cf'
        self.init_metrics = True
        self.subset_size = 1.0
        self.net_type = 'rnn'
        self.net_type_D = 'rnn'
        
        ### Architecture params
        
        # Embedders
        self.low_pass_filtering = 2
        self.base_nodes_linear = 1024
        self.depth_linear = 2
        self.norm_linear = 'bn'
        self.dropout_linear = 0.0
        
        # Encoder LSTM
        self.input_size = 136
        self.hidden_size = 1024
        self.hidden_size_D = 1024
        self.layer_norm = False
        self.zero_init = True
        
        # Decoder
        self.seq_len = 40
        self.supervision_prop = 0.0
        self.norm_deep_out = 'ln'
        self.spectral_norm = True
        self.noise_prop = 0
        self.attention = False
        self.skip_n_annotations = 2
        self.annotation_size = 60
        self.learn_init_state = True
        self.pool_in_do = False
        self.no_input = False
        self.init_mlp_coeff = 2
        self.separate_kp_jac = False
        self.dropout_initial_mlp = 0
        self.data_random_permute = False
        self.data_random_permute_D = False
        self.init_h_inpt = False
        self.init_frame_inpt = False
        self.n_blocks = 6
        self.n_heads = 8
        self.n_blocks_D = 6
        self.n_heads_D = 8
        self.residual = True
        self.position_token_dim = 0
        self.pos_encoding = True
        self.forward_mask = True
        self.autoreg_train = True
        self.temp_mixing = False
        self.emb_dim = None
        self.emb_dim_D = None
        self.pool_type = None
        self.delta_mod = False
        
        # Pooler
        self.inhibit_pool = False
        self.same = False
        self.simple_h_switch = False
        self.batch_pool = False
        self.batch_pool_G = False
        self.proj_score_y = False

        ### Losses
        
        self.adv_loss_weight = 1.0
        self.sup_loss_weight = 0.01
        self.reconstruction_loss_weight = 1e-3
        self.gan_loss = 'hinge'
        self.gp_weight = 0.1
        self.pooling_weight = 1.0
        self.seq_stream_weight = 1.0
        self.landmark_realism_weight = 1.0
        self.sync_loss = 0
        self.sync_length = 60
        self.anneal_sync_loss = False

        ### Discriminator params

        self.window = 5
        self.cnn_dis = False
        self.cnn_config = 1
        self.interaction_config = 1
        self.proj_dis_type = 'default'
        self.denser_strides = False
        self.dual_stream = True
        self.multiple_proj_dis = False
        self.length_encoding = False
        self.multiple_lstm = False
        self.normalize_dis_input = False
        self.random_translate_dis_input = False
        self.random_rotate_dis_input = False
        self.normalize_head_orientation = False
        self.separate_ldk_classif = False
        self.pos_weight = 0.5
        self.vel_weight = 0.3
        self.separate_lstm = False
        self.random_permute_D_seq = False
        self.cnn_block_depth = 3
        self.cnn_depth = 3
        self.double_seq_stream = False
        
        ### Learning params
        
        self.n_epochs = 600
        self.batch_size = 50
        self.validation_prop = 0
        self.gradient_clipping_value = 1.0
        self.n_gen_steps = 1
        self.diversity_loss_k = 1
        self.random_start_idx = False
        self.valid_index = 32
        
        # Optimizers
        self.optimizer = 'adam'
        self.learning_rate_g = 2e-5
        self.learning_rate_d = 1e-5
        self.adam_beta_1 = 0.5
        self.min_lr = 1e-8
        
        # Schedulers
        self.scheduler_type = 'step'
        self.gamma_lr = 1.0
        self.gamma_seq_len = 10
        self.gamma_supervision_prop = 0.8
        self.step_epoch_lr = 2000
        self.step_epoch_lr_2 = 3000
        self.step_epoch_lr_3 = 4000
        self.step_epoch_seq_len = 100
        self.pooling_w_converge_step = 20000 # step number
        self.kl_warmup_range = 200 # step number
        self.scheduled_sampling = False
        self.warmup_ep = 0
        self.t_max_cosinelr = 5000

        ## Disc subsequence length
        # TODO: Careful, self.seq_len will always be 40
        self.windows = {
            '0': [3, 5, 8],
            '1': [5, int(self.seq_len / 4), int(self.seq_len / 2), self.seq_len],
            '2': [3, 5, int(self.seq_len / 5), int(self.seq_len / 4), int(self.seq_len / 3), int(self.seq_len / 2), int(self.seq_len * 2 / 3), self.seq_len],
            '3': [int(self.seq_len / 2), int(self.seq_len * 2 / 3), self.seq_len],
            '4': [8],
            '5' : [3],
            '6': [3, 4, 5, 6, 7, 8, 9, 10],
            '7': [16, 18, 20, 22, 24, 26],
            '8': [4, 6, 18, 20, 34, 36],
            '9': np.arange(3, 40, 3),
            '40': [self.seq_len],
            '41': [int(self.seq_len * 2 / 3)],
            '42': [int(self.seq_len / 2)],
            '60': [3, 5, 8, 10, 13, 20, 27, 40],
            '61' :[5, 10, 20, 40],
            '100': []
        }
        self.interaction_windows = {
            '0': [3, 5, 8],
            '1': [5, int(self.seq_len / 4), int(self.seq_len / 2), self.seq_len],
            '2': [3, 5, int(self.seq_len / 5), int(self.seq_len / 4), int(self.seq_len / 3), int(self.seq_len / 2), int(self.seq_len * 2 / 3), self.seq_len],
            '3': [int(self.seq_len / 2), int(self.seq_len * 2 / 3), self.seq_len],
            '4': [8],
            '5' : [3],
            '40': [self.seq_len],
            '41': [5, 10, 20, 40],
            '42': [int(self.seq_len / 2)],
            '60': [3, 5, 8, 12, 20, 30, 40, 60],
            '61' :[3, 5, 8, 12, 20, 40],
            '100': []
        }


# Utils

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if (classname.find('Linear') != -1) & (classname.find('LinearLayer') == -1):
        nn.init.kaiming_normal_(m.weight)
    if (classname.find('Conv2d') != -1):
        nn.init.kaiming_normal_(m.weight)
    if (classname.find('Conv1d') != -1):
        nn.init.kaiming_normal_(m.weight)
        

def mix_tensors(a, b, rate):
    '''
    Replaces rate % of tensor a entries by tensor b's along first dim.
    '''
    length = len(a)
    indices = list(range(length))
    indices = np.sort(np.random.choice(indices, int(rate*length), replace=False))

    c = a.clone()
    c[indices] = b[indices]
    
    return c

def warmup_learning_rate(optimizer, nb_epo, warmup_epo, max_lr):
    """warmup learning rate"""
    lr = max_lr * nb_epo / warmup_epo
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_scheduler(obj, config, s_type, attr=None, iteration=-1):
    
    if s_type == 'lr':
        gamma = config.gamma_lr
        if config.scheduler_type == 'step':
            step_size = config.step_epoch_lr
            scheduler = torch.optim.lr_scheduler.StepLR(obj, step_size, gamma, last_epoch=iteration)
        elif config.scheduler_type == 'multistep':
            milestones = [config.step_epoch_lr, config.step_epoch_lr_2, config.step_epoch_lr_3]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(obj, milestones, gamma, last_epoch=iteration)
        elif config.scheduler_type == 'exp':
            gamma = np.exp(np.log(gamma) / config.step_epoch_lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(obj, gamma, last_epoch=iteration)
        elif config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(obj, T_max=config.t_max_cosinelr, last_epoch=iteration)
        
    elif s_type == 'obs_length':
        step_size = config.step_epoch_seq_len
        gamma = -1
        scheduler = Scheduler(obj, attr, config, step_size, gamma, last_epoch=iteration, schedule_type='incremental',
                             min_value=0)
        
    elif s_type == 'sequence_length':
        step_size = config.step_epoch_seq_len
        gamma = config.gamma_seq_len
        scheduler = Scheduler(obj, attr, config, step_size, gamma, last_epoch=iteration, schedule_type='incremental',
                             max_value=config.max_seq_len)
        
    elif s_type == 'interaction_w':
        step_size = 1
        init_value = config.pooling_weight
        final_value = config.final_pooling_weight
        gamma = np.exp((1/config.pooling_w_converge_step)*np.log(final_value/init_value))
        scheduler = Scheduler(obj, attr, config, step_size, gamma, last_epoch=iteration, schedule_type='multiplicative', max_value=final_value)

    elif s_type == 'gs_tau':
        step_size = 1
        init_value = config.tau
        final_value = config.final_tau
        gamma = (final_value - init_value)/config.tau_converge_step # linear decrease of Gumbel Softmax tau following https://arxiv.org/pdf/1611.01144.pdf
        scheduler = Scheduler(obj, attr, config, step_size, gamma, last_epoch=iteration, schedule_type='incremental',
                             min_value=final_value)
    
    elif s_type == 'kl_warmup':
        step_size = 1
        init_value = 0.0
        final_value = 1.0
        gamma = (final_value - init_value) / config.kl_warmup_range
        scheduler = Scheduler(obj, attr, config, step_size, gamma, last_epoch=iteration, schedule_type='incremental',
                             max_value=final_value)
        
    else:
        return NotImplementedError(f'Unknown scheduler type {s_type}')
    
    return scheduler


class Scheduler:
    
    def __init__(self, obj, attr, config, step_size, gamma=None, last_epoch=-1, schedule_type='incremental', min_value=0,
                 max_value=9999, min_threshold=None):
        
        self.obj = obj
        self.attr = attr
        self.init_value = getattr(config, attr)
        
        self.step_size = step_size
        self.gamma = gamma
        self.schedule_type = schedule_type
        self.min_value = min_value
        self.max_value = max_value
        self.min_threshold = min_threshold
        self.last_epoch = last_epoch + 1
        
        
    def step(self):

        self.last_epoch += 1
        
        step_number = self.last_epoch // self.step_size
            
        if self.schedule_type == 'multiplicative':
            new_val = (self.gamma ** step_number) * self.init_value
            if (self.min_threshold is not None) and (new_val < self.min_threshold):
                new_val = self.min_value

        elif self.schedule_type == 'incremental':
            new_val = self.gamma * step_number + self.init_value
            
        elif self.schedule_type == 'threshold':
            assert type(self.init_value) == bool, 'A boolean is required for this scheduler type'
            
            if self.last_epoch >= self.step_size:
                setattr(self.obj, self.attr, not self.init_value)
                return not self.init_value
            return self.init_value
            
        else:
            return NotImplementedError(f'Unknown schedule type: {self.schedule_type}')

        if type(self.init_value) == int:
            new_val = int(new_val)

        if new_val <= self.min_value:
            new_val = self.min_value
        elif self.max_value <= new_val:
            new_val = self.max_value
        
        setattr(self.obj, self.attr, new_val)
        
        return new_val


def fid(incep_pred, incep_gt, eps=1e-6):

    length = incep_pred.shape[0]

    mu_pred = incep_pred.mean(axis=0)
    cov_pred = np.matmul((incep_pred - mu_pred).T, (incep_pred - mu_pred))/length

    mu_gt = incep_gt.mean(axis=0)
    cov_gt = np.matmul((incep_gt - mu_gt).T, (incep_gt - mu_gt))/length

    # Matrix square root, stable implementation @https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

    sqrtm, _ = linalg.sqrtm(cov_pred.dot(cov_gt), disp=False)

    # Product might be almost singular
    if not np.isfinite(sqrtm).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(cov_pred.shape[0]) * eps
        sqrtm = linalg.sqrtm((cov_pred + offset).dot(cov_gt + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(sqrtm):
        if not np.allclose(np.diagonal(sqrtm).imag, 0, atol=1e-3):
            m = np.max(np.abs(sqrtm.imag))
            print('Imaginary component {}'.format(m))
        sqrtm = sqrtm.real
    
    fid = (mu_gt - mu_pred).dot((mu_gt - mu_pred)) + np.trace(cov_gt + cov_pred - 2*sqrtm)
    
    return fid


def l2_loss(x, y, dim_input=2, reduce=True):
    if dim_input > 1:
        last_dim = x.size(-1)
        x = x.view(-1, int(last_dim / dim_input), dim_input)
        y = y.view(-1, int(last_dim / dim_input), dim_input)
    if reduce:
        return torch.mean(torch.sum((x - y)**2, dim=-1))
    else:
        return torch.mean(torch.sum((x - y)**2, dim=-1), dim=-1)


def euclidean(vect, base_vects):
    size = list(vect.size())
    vect = vect.view(-1, size[-1])
    return_size = size
    return_size[-1] = 1
    
    closest_indices = torch.sum((base_vects - vect.unsqueeze(1))**2, dim=-1).argsort()[:, 0]
    return closest_indices.view(return_size)


def cosine(vect, base_vects):
    b, L, e_dim = vect.size()
    vect = vect.view(-1, e_dim)
    
    norms = torch.diag(torch.matmul(base_vects, base_vects.T))**0.5
    cosine_sim = (torch.matmul(vect, base_vects.T)/norms)/(torch.norm(vect, dim=-1).unsqueeze(1))
    return cosine_sim.argsort()[:, -1].view(b, L, 1)



def cov(mat):
    m = mat - mat.mean(dim=0)
    fact = mat.shape[0] - 1
    cov_mat = torch.matmul(torch.transpose(m, 0, 1), m)/fact
    
    return cov_mat


def batch_covariance(t, lengths):
    indexes = np.cumsum([0] + lengths)
    cov_list = []
    for i_0, i_f in zip(indexes[:-1], indexes[1:]):
        cov_list.append(cov(t[i_0:i_f, :]).unsqueeze(0))
    batch_cov = torch.cat(cov_list, dim=0)
        
    return batch_cov


def extract_diagonals(batch_cov_tensor, n_terms=4, lengths=None):
    covariances = []
    for i in range(n_terms):
        covariances.append(torch.diagonal(batch_cov_tensor, offset=i, dim1=1, dim2=2))
    
    out = torch.cat(covariances, dim=1)
    if lengths is not None:
        out = out.repeat_interleave(repeats=torch.Tensor(lengths).long().cuda(), dim=0)
        
    return out


def generate_noise(vect, lengths, noise_prop):
    
    with torch.no_grad():
        size = (len(lengths), vect.size(-1))
        std = vect.std(dim=0) * noise_prop
        noise = torch.randn(vect.size()).cuda().mul_(std)

    return noise


def gradient_penalty(inputs, outputs):
    grad = torch.autograd.grad(outputs=outputs, inputs=inputs,
                                grad_outputs=torch.ones_like(outputs).cuda(), create_graph=True, retain_graph=True)[0]
    grad = grad.contiguous().flatten(start_dim=1)
    grad_penalty = torch.mean((grad.norm(dim=1) - 1)**2)

    return grad_penalty


def bool_parser(string):
    if string.lower() == 'false':
        return False
    return True


def compute_metrics(config, solver, dataset, reps=8):

    if solver.dynamical_model.seq_len == 0:
        return 0

    all_data_loader = DataLoader(dataset, batch_size=400, collate_fn=collate_fn, shuffle=True)

    # Compute metrics on a single large batch
    batch = iter(all_data_loader).next()
    dist_to_gt = []
    with torch.no_grad():
        for i in range(reps):
            _, prediction, ground_truth, conditioning, kl = solver(batch)
            dist_to_gt.append(torch.abs((prediction - ground_truth)).sum(dim=(1, 2)))
        dist_to_gt = torch.stack(dist_to_gt)
        _, bs = dist_to_gt.size()
        min_dist_to_gt = (dist_to_gt.min(dim=0)[0].sum().item() ** 0.5) / bs

    return min_dist_to_gt



def save_metrics(checkpoint_dir, min_dist_to_gt):
    
    log_file_path = os.path.join(checkpoint_dir, 'log_file')
    with open(log_file_path, 'rb') as f:
        log_dict = pickle.load(f)
    
    log_dict['metrics']['minimum_distance'].append(min_dist_to_gt)
        
    with open(log_file_path, 'wb') as f:
        pickle.dump(log_dict, f)


def train_test_split(length, val_prop):
    indices = np.arange(length)
    np.random.shuffle(indices)
    cut = int(val_prop * length)
    return indices[cut:], indices[:cut]


def collate_fn(list_batch, to_images=False):

    lengths = [elt[0].size(0) for elt in list_batch]
    file_names = [elt[-1] for elt in list_batch]

    if to_images:
        return list_batch + [lengths] + [file_names]

    else:
        batch = [torch.cat([elt[i] for elt in list_batch], dim=0) for i in range(len(list_batch[0]) - 1)]
        return batch + [lengths] + [file_names]


def collate_vox_lips(list_batch):
    lengths = [elt[0].size(0) for elt in list_batch]
    batch = [torch.cat([elt[i] for elt in list_batch], dim=0) for i in range(len(list_batch[0]))]
    return batch + [lengths]


def length_encoding(length, dim):
    len_enc_inpt = length / (1e3 ** torch.Tensor(np.arange(dim) / dim))
    sin = torch.sin(len_enc_inpt)
    cos = torch.cos(len_enc_inpt)
    return torch.cat([sin, cos])


def merge_and_random_split(first_tensor, second_tensor, random_low=None, random_high=None, index=None):

    merged = torch.cat([first_tensor, second_tensor], dim=1)

    if index is None:
        random_observation_length = np.random.randint(random_low, random_high)
    else:
        random_observation_length = index
    
    return merged[:, :random_observation_length, ...].contiguous(), merged[:, random_observation_length:, ...].contiguous(), random_observation_length


def sample_exponential(x, min_value, max_value, decay_factor=20):
    '''
    x should be sampled from a standard uniform distribution
    '''
    l = 1 / decay_factor
    bias = np.exp(-l * min_value)
    K = bias - np.exp(-l * max_value)
    y = -np.log(-K * x + bias) / l
    return y


def ma(a, n, axis=1):
    if n == 0:
        return a
    if axis == 1:
        '''
        Moving average on axis 1
        '''
        b = torch.cumsum(a, dim=1)
        b[:, n:] = b[:, n:] - b[:, :-n]
        return b[:, n - 1:] / n
    elif axis == 0:
        '''
        Moving average on axis 0
        '''
        b = torch.cumsum(a, dim=0)
        b[n:] = b[n:] - b[:-n]
        return b[n - 1:] / n

class Arguments():
    pass
def get_default_args():
    # Params
    attr = dict(
        position = True,
        velocity = True,
        acceleration = True,
        embed_inputs = False,
        velocity_ma_range = 3,
        position_encoding_dim = 256,
        max_pool = False,
        n_heads = 5,
        transfo_dim = 256,
        n_blocks = 1,
        classif_type = 'transformer',
        time_convolve=False,
        sum_pos_emb_out=False,
        mouth = False
    )
    args = Arguments()
    for attr_name, attr_val in attr.items():
        setattr(args, str(attr_name), attr_val)
    return args


def imagify(ldm_coord_arr, shape):
    '''
    Input
    -----
    ldm_coord_arr: numpy array containing landmarks coordinates, shape: b, 68, 2
    
    Output
    -----
    landmark_list: landmarks painted as images, shape: b, 3, h, w
    '''
    
    landmark_list = []
    
    for i in range(len(ldm_coord_arr)):
        
        preds = ldm_coord_arr[i]

        input = torch.rand(shape, shape, 3) 
        dpi = 100
        fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(np.ones(input.shape))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        #chin
        ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
        #left and right eyebrow
        ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
        ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
        #nose
        ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
        ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
        #left and right eye
        ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
        ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
        #outer and inner lip
        ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
        ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
        ax.axis('off')

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        landmark_list.append(data)
        plt.close(fig)
        
    landmark_list = np.array(landmark_list)
    
    return landmark_list
    

def cal_ssim(img1, img2):
    '''
    img1 and img2 are C x H x W dimensional, no batch dimension is expected
    '''

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.clone().data.permute(1, 2, 0).cpu().numpy()
    img2 = img2.clone().data.permute(1, 2, 0).cpu().numpy()

    # Max channel dim ~500
    if img1.shape[-1] > 500:
        indices = np.random.choice(img1.shape[-1], 500, replace=False)
        img1 = img1[..., indices]
        img2 = img2[..., indices]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def cal_fid(mu_pred, cov_pred, mu_gt, cov_gt, eps=1e-6):

    # Matrix square root, stable implementation @https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

    sqrtm, _ = linalg.sqrtm(cov_pred.dot(cov_gt), disp=False)

    # Product might be almost singular
    if not np.isfinite(sqrtm).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(cov_pred.shape[0]) * eps
        sqrtm = linalg.sqrtm((cov_pred + offset).dot(cov_gt + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(sqrtm):
        if not np.allclose(np.diagonal(sqrtm).imag, 0, atol=1e-3):
            m = np.max(np.abs(sqrtm.imag))
            print('Imaginary component {}'.format(m))
        sqrtm = sqrtm.real
    
    fid = (mu_gt - mu_pred).dot((mu_gt - mu_pred)) + np.trace(cov_gt + cov_pred - 2*sqrtm)
    
    return fid


def cal_psnr(pred, target, pixel_max_cnt=255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p


class Metrics(nn.Module):
    
    def __init__(self, config, data_statistics_path=os.path.join(os.getcwd(), 'data_statistics')):
        super(Metrics, self).__init__()
        
        self.config = config
        i3d_pretrained_path = config.i3d_path
        inception_path = config.iv3_path
        
        # Load I3d for Fréchet Video Distance
        self.i3d = InceptionI3d(400, in_channels=3)
        self.i3d.load_state_dict(torch.load(i3d_pretrained_path))
        self.i3d.cuda()
        self.i3d.eval()
        for param in self.i3d.parameters():
            param.requires_grad_(False)

        # Load Iv3 for Fréchet Inception Distance
        self.iv3 = InceptionV3(inception_path).cuda()
        self.iv3.eval()
        for param in self.iv3.parameters():
            param.requires_grad_(False)
        
        self.kp = config.keypoints
        if self.kp:
            prefix = 'kp_'
        else:
            prefix = ''

        # Faster FVD metrics on sparse data points
        with open(os.path.join(data_statistics_path, f'{prefix}fvd_cov_40'), 'rb') as file:
            self.fvd_data_cov = pickle.load(file)
        with open(os.path.join(data_statistics_path, f'{prefix}fvd_mean_40'), 'rb') as file:
            self.fvd_data_mean = pickle.load(file)

        # Metrics for temporal FID
        with open(os.path.join(data_statistics_path, f'{prefix}fid_temp_cov_40'), 'rb') as file:
            self.tfid_data_cov = pickle.load(file)
        with open(os.path.join(data_statistics_path, f'{prefix}fid_temp_mean_40'), 'rb') as file:
            self.tfid_data_mean = pickle.load(file)

        # Metrics for frame FID
        with open(os.path.join(data_statistics_path, f'{prefix}fid_frame_cov'), 'rb') as file:
            self.ffid_data_cov = pickle.load(file)
        with open(os.path.join(data_statistics_path, f'{prefix}fid_frame_mean'), 'rb') as file:
            self.ffid_data_mean = pickle.load(file)

            
    def compute_statistics(self, activations):
        '''
        Input
        -----
        activations: numpy array
        '''
        mean = np.mean(activations, axis=0)
        cov = np.matmul((activations - mean).T, (activations - mean)) / (activations.shape[0] - 1)
        return mean, cov


    def inception_forward(self, x, model, split_size=20):
        features = []
        splits = x.split(split_size)
        for split in splits:
            with torch.no_grad():
                if model == 'i3d':
                    features.append(self.i3d(split.cuda()).flatten(start_dim=1).cpu().numpy())
                elif model == 'iv3':
                    features.append(self.iv3(split.cuda())[0].squeeze().cpu().numpy())
        return np.concatenate(features)


    def seq_2_img(self, seq, resolution, bs_max=20, seq_len_max=25):
        '''
        input shape : B, t, 136 * 3 (pos, vel, acc)
        output shape: B, 3, t, H, W
        '''
        bs, length, _ = seq.shape
        in_tens = seq.view(bs, length, -1, 2)[:bs_max, -seq_len_max:, :68]
        in_tens = (in_tens.flatten(end_dim=1).cpu().numpy() * resolution).astype(int)
        # From coord to images
        img_seq = imagify(in_tens, shape=resolution)
        img_seq = torch.Tensor(img_seq).transpose(1, 3)
        
        return img_seq.view(min(bs, bs_max), min(length, seq_len_max), 3, resolution, resolution).transpose(1, 2)


    def fast_seq2img(self, coord_tens, resolution=224):
        bs, length = coord_tens.shape[0], coord_tens.shape[1]
        # Convert to numpy
        coord_tens = (coord_tens.view(bs, length, -1, 2) * resolution).type(torch.long)
        # Instantiate empty image
        face_img = torch.zeros((bs, 3, length, resolution, resolution))
        # Chin for chan 1, eyes + nose for chan 2, mouth for chan 3
        chan_idx = [0, 17, 48, 68]
        for i in range(3):
            start_idx, end_idx = chan_idx[i], chan_idx[i + 1]
            if self.kp:
                start_idx, end_idx = 0, 10
            face_img[torch.arange(bs)[:, None, None], i, torch.arange(length)[None, :, None], coord_tens[..., start_idx: end_idx, 1],
                    coord_tens[..., start_idx: end_idx, 0]] = 1
        return face_img

    
    def forward(self, prediction, m_length=40):

        if self.kp:
            bs, l, _ = prediction.shape
            data_dim = int(self.config.input_size / 10)
            img_prediction = self.fast_seq2img((0.5 * (prediction.view(bs, l, -1, data_dim)[:, -40:, :10, :2] + 1)).clamp(min=0, max=0.99))
        else:
            img_prediction = self.fast_seq2img(prediction[:, -m_length:, :136].clamp(min=0, max=0.99))

        # FVD
        activations = self.inception_forward(img_prediction, 'i3d', split_size=10)
        mean, cov = self.compute_statistics(activations)
        fvd = cal_fid(mean, cov, self.fvd_data_mean, self.fvd_data_cov)

        # FID on single frames
        fid_frame_inp = img_prediction.transpose(1, 2).flatten(end_dim=1) # from bs, 3, 40, h, w to bs * 40, 3, h, w
        activations = self.inception_forward(fid_frame_inp, 'iv3', 150)
        mean, cov = self.compute_statistics(activations)
        ffid = cal_fid(mean, cov, self.ffid_data_mean, self.ffid_data_cov)

        # FID on temporal sum
        coeffs = torch.linspace(0.2, 0.8, m_length)
        sum_coef = coeffs.sum()
        coeffs = (coeffs / sum_coef).view(1, 1, m_length, 1, 1)
        fid_seq_inp = (img_prediction * coeffs).sum(dim=2)
        activations = self.iv3(fid_seq_inp.cuda())[0].squeeze().cpu().numpy()
        mean, cov = self.compute_statistics(activations)
        tfid = cal_fid(mean, cov, self.tfid_data_mean, self.tfid_data_cov)

        return fvd, ffid, tfid

        # img_ground_truth = self.seq_2_img(ground_truth, resolution)
        # img_prediction = self.seq_2_img(prediction, resolution)

        # ssim = cal_ssim(img_ground_truth.flatten(end_dim=2), img_prediction.flatten(end_dim=2))
        # psnr = cal_psnr(img_ground_truth, img_prediction)

        # if prediction.shape[1] >= 25:
        #     # norm_fvd
        #     norm_prediction = img_prediction / 255
        #     activations = self.get_inception_activation(norm_prediction)
        #     mean, cov = self.compute_statistics(activations.squeeze())
        #     norm_fvd = cal_fid(mean, cov, self.data_mean_norm, self.data_cov_norm)
        #     # unnormed
        #     activations = self.get_inception_activation(img_prediction)
        #     mean, cov = self.compute_statistics(activations.squeeze())
        #     fvd = cal_fid(mean, cov, self.data_mean, self.data_cov)
        # else:
        #     norm_fvd = 0
        #     fvd = 0
        # return norm_fvd, fvd, ssim, psnr


## Rotation & translation functions for batched Tensors

def get_R_matrix_from_tensor(tens):
    '''
    Returns rotation matrix that frontalizes first face image in a sequence
    params:
    ------
    tens: Tensor of shape bs, seq_len, 68, 3
    '''
    # Project on y axis and normalize
    tens = tens.cpu()
    proj_y = (tens[:, 0, 0] - tens[:, 0, 16])[:, [0, 2]]
    sign = (proj_y[:, 0] / torch.abs(proj_y[:, 0])).unsqueeze(-1)
    proj_y = sign * proj_y / torch.norm(proj_y, dim=-1).unsqueeze(1)
    
    sin_half = (proj_y[:, 1] / torch.abs(proj_y[:, 1])) * ((0.5 * (1 - proj_y[:, 0])) ** 0.5)
    cos_half = (0.5 * (1 + proj_y[:, 0])) ** 0.5
    ry = torch.Tensor(np.array([R.from_quat([0, sin_half[i], 0, cos_half[i]]).as_matrix() for i in range(len(sin_half))])).cuda()
    return ry

def T_matrix_from_tensor(origin):
    '''
    origin of shape bs, 3
    '''
    return torch.cat([torch.eye(3).repeat(origin.shape[0], 1, 1).cuda(), origin.unsqueeze(-1)], dim=-1)

def translate_tensor(vector, M):
    ones = torch.ones_like(vector)[..., [0]].cuda()
    return torch.bmm(torch.cat([vector, ones], dim=-1).flatten(start_dim=1, end_dim=2),
          M.transpose(1, 2)).view(vector.shape)

def b_rotate_3D_tensor(tensor, sin_half=None, cos_half=None):
    '''
    tens: Tensor of shape bs, seq_len, 68 * 3 (dim) * 3 (x, v, a)
    '''

    # Center position tensor
    bs, seq_len, input_dim = tensor.shape
    positions = tensor[..., :68 * 3].contiguous().view(bs, seq_len, 68, 3)
    origin = positions.mean(dim=(1, 2))
    centerred_tensor = translate_tensor(positions, T_matrix_from_tensor(-origin))

    # Construct rotation matrix bs * 3 * 3
    if sin_half is None:
        with torch.no_grad():
            rotation_matrix = get_R_matrix_from_tensor(centerred_tensor)
    else:
        rotation_matrix = torch.Tensor(np.array([R.from_quat([0, sin_half[i], 0, cos_half[i]]).as_matrix() for i in range(len(sin_half))])).cuda()

    rotated_centerred_tensor = torch.bmm(centerred_tensor.flatten(start_dim=1, end_dim=2),
              rotation_matrix.transpose(1, 2)).view(centerred_tensor.shape)
    rotated_tensor = translate_tensor(rotated_centerred_tensor, T_matrix_from_tensor(origin))

    out = [rotated_tensor.flatten(start_dim=-2)]

    # Apply rotation on velocity and acceleration
    for i in range(1, tensor.shape[-1] // (3 * 68)):
        split = tensor[..., i * 3 * 68: (i + 1) * 3 * 68].contiguous().view(bs, seq_len, 68, 3)
        out.append(torch.bmm(split.flatten(start_dim=1, end_dim=2), rotation_matrix.transpose(1, 2)).view(bs, seq_len, -1))

    return torch.cat(out, dim=-1)

###

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # args.distributed = False
    # return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def is_main_process():
    return dist.get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
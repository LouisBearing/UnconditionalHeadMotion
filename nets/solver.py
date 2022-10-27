from .networks import *
from .utils import *
import numpy as np
import os
import torch.nn as nn
import pickle


class InteractionSolver(nn.Module):
    
    def __init__(self, config):
        super(InteractionSolver, self).__init__()
        
        self.config = config

        # General params: landmarks / keypoints, observation length
        self.keypoints = config.keypoints
        self.obs_len = config.obs_len
        
        # Loss function contributions
        self.lbda_adv_loss = config.adv_loss_weight
        self.lbda_sup_loss = config.sup_loss_weight
        self.lbda_reco_loss = config.reconstruction_loss_weight
        
        # Architecture preferences
        self.layer_norm = config.layer_norm
        self.gradient_clipping_value = config.gradient_clipping_value
        
        ## Dynamical Model
        if config.net_type == 'rnn':
            self.dynamical_model = DynamicalModel(config)
        elif config.net_type == 'transformer':
            self.dynamical_model = TransformerGenerator(config)
        elif config.net_type == 'dvae':
            self.dynamical_model = SRNNGenerator(config)
        self.dynamical_model.cuda()

        ## Discriminator
        self.dis = Discriminator(config)
        self.dis.cuda()
        
        ## Optimizers
        betas = (config.adam_beta_1, 0.999)
        self.optim_D_params = list(self.dis.parameters())
        self.optim_G_params = list(self.dynamical_model.parameters())

        if config.warmup_ep > 0:
            lr_d = config.min_lr
            lr_g = config.min_lr
        else:
            lr_d = config.learning_rate_d
            lr_g = config.learning_rate_g
        if self.config.optimizer == 'adadelta':
            self.optim_D = torch.optim.Adadelta(params=self.optim_D_params)
            self.optim_G = torch.optim.Adadelta(params=self.optim_G_params)
        elif self.config.optimizer == 'adamw':
            self.optim_D = torch.optim.AdamW(params=self.optim_D_params, betas=betas, lr=lr_d)
            self.optim_G = torch.optim.AdamW(params=self.optim_G_params, betas=betas, lr=lr_g)
        else:
            self.optim_D = torch.optim.Adam(params=self.optim_D_params, betas=betas, lr=lr_d)
            self.optim_G = torch.optim.Adam(params=self.optim_G_params, betas=betas, lr=lr_g)

        ## Metrics
        if config.init_metrics:
            self.metrics = Metrics(config)

        ## Schedulers
        if config.gamma_lr < 1:
            self.gen_scheduler = get_scheduler(self.optim_G, config, s_type='lr')
            self.dis_scheduler = get_scheduler(self.optim_D, config, s_type='lr')
        if config.scheduled_sampling:
            setattr(self.config, 'tforcing', self.config.seq_len - 1)
            self.tforcing_scheduler = get_scheduler(self.config, self.config, s_type='obs_length', attr='tforcing')
        else:
            setattr(self.config, 'tforcing', 0)
            
        ## Dictionary to track losses and scheduler params history
        self.log_dict = {}
        loss_keys = ['total', 'adversarial', 'adversarial_interaction', 'adversarial_landmarks', 'supervised', 'reconstruction', 'kl']
        for mode in ['training', 'validation']:
            self.log_dict.update({
                f'{mode}_losses': {loss: [] for loss in loss_keys}
            })
        self.log_dict.update({
            'running_losses': {loss: [] for loss in loss_keys}
        })
        self.log_dict['dis_losses'] = dict(
            individual=[],
            interaction=[],
            landmarks=[]
            )
        self.log_dict['metrics'] = dict(
            fvd=[],
            fid=[],
            tfid=[],
            minimum_distance=[]
        )
        self.log_dict['dis_out'] = dict(
            fake=[],
            real=[]
        )
        schedulers_info_keys = ['supervision_rate', 'sequence_length', 'interaction_w', 'tforcing_len']
        self.log_dict.update({key: [] for key in schedulers_info_keys})


    def forward(self, batch, shuffle=False, random_start_idx=False, seq_len=None, obs_len=None, eval_mode=False):

        if seq_len is None:
            seq_len = self.config.seq_len - self.config.tforcing
        if obs_len is None:
            obs_len = self.obs_len + self.config.tforcing
        
        # Confer
        if len(batch) == 5: # self.config.dataset == 'cf':
            sequence, spectro, bboxes, lengths, filenames = batch # lengths is a list of participants number in each interaction, bboxes is only applicable for unsup keypoints
        # Sewa
        elif self.config.dataset == 'sewa':
            sequence, lengths, filenames = batch
            sequence = sequence[:, ::2, :] # from 50hz to 25, comparable to cf & vox
        # Vox
        else:
            shuffle = True
            landmarks, spectro, seq_lengths = batch
            sequence = []
            required_len = obs_len + seq_len + self.config.velocity_ma_range + 2
            indexes = np.cumsum([0] + seq_lengths)
            for i_0, i_f in zip(indexes[:-1], indexes[1:]):
                if i_f - i_0 >= required_len:
                    sequence.append(landmarks[i_0:i_0 + required_len])
            sequence = torch.stack(sequence)
            random_start_idx = False
        
        sequence = sequence.cuda()

        if shuffle:
            perm_idx = np.arange(sequence.shape[0])
            np.random.shuffle(perm_idx)
            bs = sequence.shape[0] // self.config.n_pers
            sequence = sequence[perm_idx[:self.config.n_pers * bs]]
            lengths = [self.config.n_pers] * bs

        # Velocity
        velocities = sequence[:, 1:, ...] - sequence[:, :-1, ...]
        low_pass_velocities = ma(velocities, n=self.config.velocity_ma_range)
        sequence = torch.cat([sequence[:, max(1, self.config.velocity_ma_range):, ...], low_pass_velocities], dim=2)

        # Acceleration
        acceleration = low_pass_velocities[:, 1:, ...] - low_pass_velocities[:, :-1, ...]
        sequence = torch.cat([sequence[:, 1:, ...], acceleration], dim=2).flatten(start_dim=-2)

        # Observation / Prediction split
        if random_start_idx:
            start_idx = np.random.randint(obs_len, sequence.size(1) - obs_len - seq_len)
        else:
            start_idx = 0
        obs_seq = sequence[:, start_idx:start_idx + obs_len].contiguous()
        gt_seq = sequence[:, start_idx + obs_len:].contiguous()
        
        # Dynamical Model
        if self.config.net_type == 'dvae':
            if eval_mode:
                input_sequence = obs_seq
                generation = self.dynamical_model.train_step(input_sequence, seq_len)#TODO: à terminer...
            else:
                # In dvae training case, the full gt sequence in given as input to the autoencoding G that is trained to reconstruct it
                input_sequence = sequence[:, start_idx:start_idx + obs_len + seq_len].contiguous()
                generation = self.dynamical_model.train_step(input_sequence, seq_len)
            reconstruction = torch.empty(0, 0, 0).cuda()
            conditioning = torch.empty(0, 0, 0).cuda()
            ground_truth = input_sequence[:, 1:]
        elif self.config.net_type == 'transformer':
            # if (eval_mode == False) and (self.config.autoreg_train == False):
            #     input_sequence = sequence[:, start_idx:start_idx + obs_len + seq_len]
            #     reconstruction, generation = self.dynamical_model(input_sequence, obs_len, 0)
            # else:
            reconstruction, generation = self.dynamical_model(obs_seq, obs_len, seq_len)
            ground_truth = gt_seq[:, :seq_len, :].contiguous()
            conditioning = obs_seq[:, 1:, :].contiguous()
        else:
            reconstruction, generation = self.dynamical_model(obs_seq, lengths, seq_len)
            ground_truth = gt_seq[:, :seq_len, :].contiguous()
            conditioning = obs_seq[:, 1:, :].contiguous()

        return reconstruction, generation, ground_truth, conditioning, lengths

    
    def dis_update(self, batch, scaler=None):

        self.optim_D.zero_grad()

        # with autocast():
        with torch.no_grad():
            reconstructed_prediction, prediction, ground_truth, conditioning, lengths = self.forward(batch, shuffle=self.config.data_random_permute_D, 
                random_start_idx=self.config.random_start_idx)
        
        if self.lbda_adv_loss == 0:
            return torch.tensor(0.0).cuda()

        (adv_loss, loss_individual, loss_interaction, loss_landmarks, all_f_out, all_r_out) = self.dis.compute_dis_loss(prediction, ground_truth, conditioning, lengths)
        
        if scaler is not None:
            scaler.scale(adv_loss).backward()
            if self.gradient_clipping_value > 0:
                scaler.unscale_(self.optim_D)
                nn.utils.clip_grad_norm_(self.optim_D_params, self.gradient_clipping_value)
            scaler.step(self.optim_D)
            scaler.update()
        else:
            adv_loss.backward()
            if self.gradient_clipping_value > 0:
                nn.utils.clip_grad_norm_(self.optim_D_params, self.gradient_clipping_value)
            self.optim_D.step()
        
        self.log_dict['dis_losses']['individual'].append(loss_individual.item())
        self.log_dict['dis_losses']['interaction'].append(loss_interaction.item())
        self.log_dict['dis_losses']['landmarks'].append(loss_landmarks.item())
        self.log_dict['dis_out']['fake'].append(all_f_out)
        self.log_dict['dis_out']['real'].append(all_r_out)

        print('Dis updated!')
        
        return adv_loss
        
    
    def gen_update(self, batch, train=True, writer=None, scaler=None, compute_metrics=False, counter=0):
        self.optim_G.zero_grad()

        # Start autocast
        # with autocast():
        reconstructed_prediction, prediction, ground_truth, conditioning, lengths = self.forward(batch, shuffle=self.config.data_random_permute, 
            random_start_idx=self.config.random_start_idx)

        out = ()
        # Adversarial loss
        if self.lbda_adv_loss == 0:
            adv_loss = torch.tensor(0.0).cuda()
            loss_individual = torch.tensor(0.0).cuda()
            loss_interaction = torch.tensor(0.0).cuda()
            loss_landmarks = torch.tensor(0.0).cuda()
        else:
            (adv_loss, loss_individual, loss_interaction, loss_landmarks) = self.dis.compute_gen_loss(prediction, ground_truth, conditioning, lengths)

        loss = self.lbda_adv_loss * adv_loss
        out += (adv_loss.item(),)
        
        # Supervised loss
        if (self.config.keypoints == True) and (self.config.jacobian == True):
            dim_input = 6
        else:
            dim_input = 2

        if self.config.net_type == 'dvae':
            prediction, sigma, kl = prediction
            kl = kl.mean()
            loss += self.lbda_sup_loss * kl
            supervised_loss = 0.5 * (((ground_truth[..., :self.config.input_size] - prediction[..., :self.config.input_size]) / sigma) ** 2)
            supervised_loss = supervised_loss.sum(dim=-1).mean()
            reconstruction_loss = torch.tensor(0.0).cuda()
        else:
            # supervised_loss = l2_loss(prediction[..., :self.config.input_size], ground_truth[..., :self.config.input_size], dim_input=dim_input)
            # reconstruction_loss = l2_loss(reconstructed_prediction[..., :self.config.input_size], conditioning[..., :self.config.input_size], dim_input=dim_input)

            supervised_loss = l2_loss(prediction, ground_truth, dim_input=dim_input)
            reconstruction_loss = l2_loss(reconstructed_prediction, conditioning, dim_input=dim_input)
            
        loss += self.lbda_sup_loss * supervised_loss + self.lbda_reco_loss * reconstruction_loss
        out += (supervised_loss.item(),)
        
        out = (loss.item(),) + out
        ## End autocast

        print(f'Gen updated, loss : {out}')
        
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if self.gradient_clipping_value > 0:
                    scaler.unscale_(self.optim_G)
                    nn.utils.clip_grad_norm_(self.optim_G_params, self.gradient_clipping_value)
                scaler.step(self.optim_G)
                scaler.update()
            else:
                loss.backward()
                if self.gradient_clipping_value > 0:
                    nn.utils.clip_grad_norm_(self.optim_G_params, self.gradient_clipping_value)
                self.optim_G.step()
            key = 'training'
        else:
            key = 'validation'
            if compute_metrics:
                # Compute metrics
                if self.config.seq_len > 0:
                    metrics_input = prediction
                else:
                    metrics_input = reconstructed_prediction
                fvd, fid, tfid = self.metrics(metrics_input)
                self.log_dict['metrics']['fvd'].append(fvd)
                self.log_dict['metrics']['fid'].append(fid)
                self.log_dict['metrics']['tfid'].append(tfid)
            
        self.log_dict[f'{key}_losses']['total'].append(loss.item())
        scaled_adv_l = self.lbda_adv_loss * loss_individual.item()
        self.log_dict[f'{key}_losses']['adversarial'].append(scaled_adv_l)
        scaled_adv_inter_l = self.lbda_adv_loss * loss_interaction.item()
        self.log_dict[f'{key}_losses']['adversarial_interaction'].append(scaled_adv_inter_l)
        scaled_adv_frame_l = self.lbda_adv_loss * loss_landmarks.item()
        self.log_dict[f'{key}_losses']['adversarial_landmarks'].append(scaled_adv_frame_l)
        scaled_mse_l = self.lbda_sup_loss * supervised_loss.item()
        self.log_dict[f'{key}_losses']['supervised'].append(scaled_mse_l)
        scaled_reco_l = self.lbda_reco_loss * reconstruction_loss.item()
        self.log_dict[f'{key}_losses']['reconstruction'].append(scaled_reco_l)

        if writer is not None:
            writer.add_scalar('Loss/adv_seq', scaled_adv_l, global_step=counter)
            writer.add_scalar('Loss/adv_inter', scaled_adv_inter_l, global_step=counter)
            writer.add_scalar('Loss/adv_frame', scaled_adv_frame_l, global_step=counter)
            writer.add_scalar('Loss/supervised', scaled_mse_l, global_step=counter)
            writer.add_scalar('Loss/reconstruction', scaled_reco_l, global_step=counter)
        

        if self.config.net_type == 'dvae':
            self.log_dict[f'{key}_losses']['kl'].append(self.lbda_sup_loss * kl.item())
        return out
    
    
    def save(self, directory, epoch, loader_length, new_file=False, save_dis=False, best=False):
        
        # Make sure gradients have not exploded
        if self.config.net_type == 'rnn':
            if self.layer_norm:
                condition = torch.isnan(self.dynamical_model.lstm.gates_h.weight[0][0]).item()
            else:
                condition = torch.isnan(self.dynamical_model.lstm.weight_hh_l0[0][0]).item()
            if condition:
                return -1

        save_dict = {
            'dynamical_model': self.dynamical_model.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'iter': epoch * loader_length,
            'epoch': epoch
        }
        if save_dis:
            save_dict.update({
                'dis': self.dis.state_dict(),
                'optim_D': self.optim_D.state_dict()
                })

        if best:
            torch.save(save_dict, os.path.join(directory, f'model_chkpt_best.pt'))
        elif new_file:
            torch.save(save_dict, os.path.join(directory, f'model_chkpt_{epoch}.pt'))
        else:
            torch.save(save_dict, os.path.join(directory, f'model_chkpt.pt'))
        
        # Log file        
        log_file_path = os.path.join(directory, 'log_file')

        with open(log_file_path, 'wb') as f:
            pickle.dump(self.log_dict, f)
        
        return 0

            
    def resume(self, checkpoint_dir, iteration=None, load_dis=True):

        search = 'model_chkpt.pt'
        if iteration is not None:
            search = f'model_chkpt_{iteration}.pt'
        file = [f for f in os.listdir(checkpoint_dir) if search in f][0]

        checkpoint = torch.load(os.path.join(checkpoint_dir, file))

        # Networks
        self.dynamical_model.load_state_dict(checkpoint['dynamical_model'])
        if load_dis:
            self.dis.load_state_dict(checkpoint['dis'])
            # Optimizers
            self.optim_D.load_state_dict(checkpoint['optim_D'])
            self.optim_G.load_state_dict(checkpoint['optim_G'])

        # Schedulers
        epoch = checkpoint['epoch']
        iteration = epoch - 1
        
        if self.config.gamma_lr < 1:
            self.gen_scheduler = get_scheduler(self.optim_G, self.config, s_type='lr', iteration=iteration)
            self.dis_scheduler = get_scheduler(self.optim_D, self.config, s_type='lr', iteration=iteration)        
        if self.config.supervision_prop > 0:
            self.supervision_scheduler = get_scheduler(self.dynamical_model, self.config, s_type='supervision_rate',
                                                       attr='supervision_prop', iteration=iteration)
        if self.config.scheduled_sampling:
            setattr(self.config, 'tforcing', self.config.seq_len - 1)
            self.tforcing_scheduler = get_scheduler(self.config, self.config, s_type='obs_length', attr='tforcing', iteration=iteration)
            
        # Log file
        log_file_path = os.path.join(checkpoint_dir, 'log_file')
        with open(log_file_path, 'rb') as f:
            self.log_dict = pickle.load(f)

        # if not 'dis_out' in self.log_dict.keys():
        #     self.log_dict['dis_out'] = dict(
        #         fake=[],
        #         real=[]
        #     )
            
        return epoch


    def step_scheduler(self):
        
        if hasattr(self, 'gen_scheduler'):
            self.gen_scheduler.step()
            self.dis_scheduler.step()
        if hasattr(self, 'supervision_scheduler'):
            supervision_rate = self.supervision_scheduler.step()
            self.log_dict['supervision_rate'].append(supervision_rate)
        if hasattr(self, 'tforcing_scheduler'):
            tforcing_len = self.tforcing_scheduler.step()
            self.log_dict['tforcing_len'].append(tforcing_len)
        if hasattr(self, 'seq_len_scheduler'):
            seq_len = self.seq_len_scheduler.step()
            self.log_dict['sequence_length'].append(seq_len)
        if hasattr(self, 'interaction_w_scheduler'):
            interaction_w = self.interaction_w_scheduler.step()
            self.log_dict['interaction_w'].append(interaction_w)
        if hasattr(self, 'kl_warmup_scheduler'):
            kl_warmup_coef = self.kl_warmup_scheduler.step()
        if self.config.anneal_sync_loss and self.config.sync_loss > 0:
            self.sync_loss = min(0.1, self.sync_loss * (100 ** (1 / 3000)))

        return self.gen_scheduler.get_last_lr()[0], self.dis_scheduler.get_last_lr()[0]

    def warmup_lr(self, epoch):
        lr_g = warmup_learning_rate(self.optim_G, epoch, self.config.warmup_ep, self.config.learning_rate_g)
        lr_d = warmup_learning_rate(self.optim_D, epoch, self.config.warmup_ep, self.config.learning_rate_d)
        return lr_g, lr_d

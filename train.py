import os
import argparse
import numpy as np
import json
from nets.utils import *
from nets.solver import *
from dataset.confer_dataset import *
from dataset.vox_lips_dataset import *
from log_plot import *
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import GradScaler


config = Config()
parser = argparse.ArgumentParser()

### General params

parser.add_argument('--model_name', default='new_model', type=str)
parser.add_argument('--data_path', default=os.getcwd(), type=str)
parser.add_argument('--save_dir', default=config.save_dir, type=str)
parser.add_argument('--save_every', default=config.save_every, type=int)
parser.add_argument('--velocity', default=config.velocity, type=bool_parser)
parser.add_argument('--acceleration', default=config.acceleration, type=bool_parser)
parser.add_argument('--velocity_input', default=config.velocity_input, type=bool_parser)
parser.add_argument('--acceleration_input', default=config.acceleration_input, type=bool_parser)
parser.add_argument('--velocity_ma_range', default=config.velocity_ma_range, type=int)
parser.add_argument('--velocity_output', default=config.velocity_output, type=bool_parser)
parser.add_argument('--acceleration_output', default=config.acceleration_output, type=bool_parser)
parser.add_argument('--positions', default=config.positions, type=bool_parser)
parser.add_argument('--data_dim', default=config.data_dim, type=int)
parser.add_argument('--obs_len', default=config.obs_len, type=int)
parser.add_argument('--keypoints', default=config.keypoints, type=bool_parser)
parser.add_argument('--jacobian', default=config.jacobian, type=bool_parser)
parser.add_argument('--fine_tuning_data', default=config.fine_tuning_data, type=bool_parser)
parser.add_argument('--range_high', default=config.range_high, type=float)
parser.add_argument('--range_low', default=config.range_low, type=float)
parser.add_argument('--optimal_setting', default=config.optimal_setting, type=bool_parser)
parser.add_argument('--mean_s', default=config.mean_s, type=float)
parser.add_argument('--std_s', default=config.std_s, type=float)
parser.add_argument('--rescaling_uniform', default=config.rescaling_uniform, type=bool_parser)
parser.add_argument('--n_pers', default=config.n_pers, type=int)
parser.add_argument('--dataset', default=config.dataset, type=str)
parser.add_argument('--subset_size', default=config.subset_size, type=float)
parser.add_argument('--net_type', default=config.net_type, type=str)
parser.add_argument('--net_type_D', default=config.net_type_D, type=str)

### Architecture params

# Embedders
parser.add_argument('--low_pass_filtering', default=config.low_pass_filtering, type=int)
parser.add_argument('--base_nodes_linear', default=config.base_nodes_linear, type=int)
parser.add_argument('--depth_linear', default=config.depth_linear, type=int)
parser.add_argument('--norm_linear', default=config.norm_linear, type=str)
parser.add_argument('--dropout_linear', default=config.dropout_linear, type=float)

# Encoder LSTM
parser.add_argument('--input_size', default=config.input_size, type=int)
parser.add_argument('--hidden_size', default=config.hidden_size, type=int)
parser.add_argument('--hidden_size_D', default=config.hidden_size_D, type=int)
parser.add_argument('--layer_norm', default=config.layer_norm, type=bool_parser)
parser.add_argument('--zero_init', default=config.zero_init, type=bool_parser)

# Decoder
parser.add_argument('--seq_len', default=config.seq_len, type=int)
parser.add_argument('--supervision_prop', default=config.supervision_prop, type=float)
parser.add_argument('--norm_deep_out', default=config.norm_deep_out, type=str)
parser.add_argument('--spectral_norm', default=config.spectral_norm, type=bool_parser)
parser.add_argument('--noise_prop', default=config.noise_prop, type=float)
parser.add_argument('--attention', default=config.attention, type=bool_parser)
parser.add_argument('--skip_n_annotations', default=config.skip_n_annotations, type=int)
parser.add_argument('--annotation_size', default=config.annotation_size, type=int)
parser.add_argument('--learn_init_state', default=config.learn_init_state, type=bool_parser)
parser.add_argument('--pool_in_do', default=config.pool_in_do, type=bool_parser)
parser.add_argument('--no_input', default=config.no_input, type=bool_parser)
parser.add_argument('--init_mlp_coeff', default=config.init_mlp_coeff, type=int)
parser.add_argument('--separate_kp_jac', default=config.separate_kp_jac, type=bool_parser)
parser.add_argument('--dropout_initial_mlp', default=config.dropout_initial_mlp, type=float)
parser.add_argument('--data_random_permute', default=config.data_random_permute, type=bool_parser)
parser.add_argument('--data_random_permute_D', default=config.data_random_permute_D, type=bool_parser)
parser.add_argument('--init_h_inpt', default=config.init_h_inpt, type=bool_parser)
parser.add_argument('--init_frame_inpt', default=config.init_frame_inpt, type=bool_parser)
parser.add_argument('--n_blocks', default=config.n_blocks, type=int)
parser.add_argument('--n_heads', default=config.n_heads, type=int)
parser.add_argument('--n_blocks_D', default=config.n_blocks_D, type=int)
parser.add_argument('--n_heads_D', default=config.n_heads_D, type=int)
parser.add_argument('--residual', default=config.residual, type=bool_parser)
parser.add_argument('--position_token_dim', default=config.position_token_dim, type=int)
parser.add_argument('--pos_encoding', default=config.pos_encoding, type=bool_parser)
parser.add_argument('--forward_mask', default=config.forward_mask, type=bool_parser)
parser.add_argument('--autoreg_train', default=config.autoreg_train, type=bool_parser)
parser.add_argument('--temp_mixing', default=config.temp_mixing, type=bool_parser)
parser.add_argument('--emb_dim', default=config.emb_dim, type=int)
parser.add_argument('--emb_dim_D', default=config.emb_dim_D, type=int)
parser.add_argument('--pool_type', default=config.pool_type, type=str)
parser.add_argument('--delta_mod', default=config.delta_mod, type=bool_parser)

# Pooler
parser.add_argument('--inhibit_pool', default=config.inhibit_pool, type=bool_parser)
parser.add_argument('--same', default=config.same, type=bool_parser)
parser.add_argument('--simple_h_switch', default=config.simple_h_switch, type=bool_parser)
parser.add_argument('--batch_pool', default=config.batch_pool, type=bool_parser)
parser.add_argument('--batch_pool_G', default=config.batch_pool_G, type=bool_parser)
parser.add_argument('--proj_score_y', default=config.proj_score_y, type=bool_parser)

### Losses

parser.add_argument('--adv_loss_weight', default=config.adv_loss_weight, type=float)
parser.add_argument('--sup_loss_weight', default=config.sup_loss_weight, type=float)
parser.add_argument('--reconstruction_loss_weight', default=config.reconstruction_loss_weight, type=float)
parser.add_argument('--gan_loss', default=config.gan_loss, type=str)
parser.add_argument('--gp_weight', default=config.gp_weight, type=float)
parser.add_argument('--pooling_weight', default=config.pooling_weight, type=float)
parser.add_argument('--sync_loss', default=config.sync_loss, type=float)
parser.add_argument('--sync_length', default=config.sync_length, type=int)
parser.add_argument('--anneal_sync_loss', default=config.anneal_sync_loss, type=bool_parser)
parser.add_argument('--seq_stream_weight', default=config.seq_stream_weight, type=float)
parser.add_argument('--landmark_realism_weight', default=config.landmark_realism_weight, type=float)

### Discriminator params

parser.add_argument('--window', default=config.window, type=int)
parser.add_argument('--cnn_dis', default=config.cnn_dis, type=bool_parser)
parser.add_argument('--cnn_config', default=config.cnn_config, type=int)
parser.add_argument('--interaction_config', default=config.interaction_config, type=int)
parser.add_argument('--proj_dis_type', default=config.proj_dis_type, type=str)
parser.add_argument('--denser_strides', default=config.denser_strides, type=bool_parser)
parser.add_argument('--dual_stream', default=config.dual_stream, type=bool_parser)
parser.add_argument('--normalize_dis_input', default=config.normalize_dis_input, type=bool_parser)
parser.add_argument('--random_translate_dis_input', default=config.random_translate_dis_input, type=bool_parser)
parser.add_argument('--random_rotate_dis_input', default=config.random_rotate_dis_input, type=bool_parser)
parser.add_argument('--normalize_head_orientation', default=config.normalize_head_orientation, type=bool_parser)
parser.add_argument('--separate_ldk_classif', default=config.separate_ldk_classif, type=bool_parser)
parser.add_argument('--pos_weight', default=config.pos_weight, type=float)
parser.add_argument('--vel_weight', default=config.vel_weight, type=float)
parser.add_argument('--separate_lstm', default=config.separate_lstm, type=bool_parser)
parser.add_argument('--random_permute_D_seq', default=config.random_permute_D_seq, type=bool_parser)
parser.add_argument('--cnn_block_depth', default=config.cnn_block_depth, type=int)
parser.add_argument('--cnn_depth', default=config.cnn_depth, type=int)
parser.add_argument('--double_seq_stream', default=config.double_seq_stream, type=bool_parser)

### Learning params

parser.add_argument('--n_epochs', default=config.n_epochs, type=int)
parser.add_argument('--batch_size', default=config.batch_size, type=int)
parser.add_argument('--validation_prop', default=config.validation_prop, type=float)
parser.add_argument('--gradient_clipping_value', default=config.gradient_clipping_value, type=float)
parser.add_argument('--n_gen_steps', default=config.n_gen_steps, type=int)
parser.add_argument('--diversity_loss_k', default=config.diversity_loss_k, type=int)
parser.add_argument('--random_start_idx', default=config.random_start_idx, type=bool_parser)

parser.add_argument('--valid_index', default=config.valid_index, type=int)

# Optimizers
parser.add_argument('--optimizer', default=config.optimizer, type=str)
parser.add_argument('--learning_rate_g', default=config.learning_rate_g, type=float)
parser.add_argument('--learning_rate_d', default=config.learning_rate_d, type=float)
parser.add_argument('--adam_beta_1', default=config.adam_beta_1, type=float)

# Schedulers
parser.add_argument('--scheduler_type', default=config.scheduler_type, type=str)
parser.add_argument('--gamma_lr', default=config.gamma_lr, type=float)
parser.add_argument('--gamma_seq_len', default=config.gamma_seq_len, type=int)
parser.add_argument('--gamma_supervision_prop', default=config.gamma_supervision_prop, type=float)
parser.add_argument('--step_epoch_lr', default=config.step_epoch_lr, type=int)
parser.add_argument('--step_epoch_seq_len', default=config.step_epoch_seq_len, type=int)
parser.add_argument('--pooling_w_converge_step', default=config.pooling_w_converge_step, type=int)
parser.add_argument('--scheduled_sampling', default=config.scheduled_sampling, type=bool_parser)
parser.add_argument('--warmup_ep', default=config.warmup_ep, type=int)
parser.add_argument('--t_max_cosinelr', default=config.t_max_cosinelr, type=int)

# Distributed
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


## Arguments parsing and config parameters setting
args = parser.parse_args()
init_distributed_mode(args)
for attr, attr_value in args.__dict__.items():
    setattr(config, attr, attr_value)
    
## Looking for a previous checkpoint
save_dir = os.path.join(config.save_dir, config.model_name)
resume = False
if os.path.isdir(save_dir):
    resume = True
else:
    condition = False
    if args.distributed:
        if is_main_process():
            condition = True
    else:
        condition = True
    if condition:
        os.mkdir(save_dir)

## Tensorboard
writer = SummaryWriter(save_dir)

# Save the configuration
args_serialize_path = os.path.join(save_dir, 'args')
with open(args_serialize_path, 'w') as f:
    json.dump(args.__dict__, f)


## Dataset instanciation
if config.dataset == 'cf':
    dataset = Confer_Dataset(config.data_path, low_pass_filtering=config.low_pass_filtering, dim=config.data_dim, keypoints=config.keypoints,
        jacobian=config.jacobian, fine_tuning_data=config.fine_tuning_data, range_high=config.range_high, range_low=config.range_low, mean_s=config.mean_s, 
        std_s=config.std_s, optimal_setting=config.optimal_setting, valid_index=config.valid_index)
    test_dataset = Confer_Dataset(config.data_path, low_pass_filtering=config.low_pass_filtering, test=True, dim=config.data_dim,
     keypoints=config.keypoints, jacobian=config.jacobian, fine_tuning_data=config.fine_tuning_data)
    collate = collate_fn
    num_workers = 0
elif config.dataset == 'vox':
    dataset = VoxLipsDataset('preprocessed_VoxCeleb', low_pass_filtering=config.low_pass_filtering, subset_size=config.subset_size)
    test_dataset = VoxLipsDataset('preprocessed_VoxCeleb', low_pass_filtering=config.low_pass_filtering, test=True)
    collate = collate_vox_lips
    num_workers = 6
    if config.subset_size < 1.0:
        if resume:
            with open(os.path.join(save_dir, 'training_samples'), 'rb') as f:
                vid_ids = pickle.load(f)
            dataset.vid_id = vid_ids
        else:
            with open(os.path.join(save_dir, 'training_samples'), 'wb') as f:
                pickle.dump(dataset.vid_id, f)
elif config.dataset == 'sewa':
    dataset = SEWA_Dataset(config.data_path, low_pass_filtering=config.low_pass_filtering)
    test_dataset = Confer_Dataset(config.data_path, low_pass_filtering=config.low_pass_filtering, test=True, keypoints=config.keypoints, jacobian=config.jacobian)
    collate = collate_fn
if config.keypoints == True:
    if config.jacobian == False:
        data_dim = 10 * 2
    else:
        data_dim = 10 * 6
else:
    data_dim = 68 * config.data_dim
setattr(config, 'input_size', data_dim)

## Train & validation sets
# train_indices, val_indices = train_test_split(len(dataset), val_prop=config.validation_prop)

if args.distributed:
    print('Distributed samplers')
    train_sampler = DistributedSampler(dataset)
else:
    train_sampler = RandomSampler(dataset)
train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate, num_workers=num_workers)
gen_iterator = iter(train_loader)

test_loader = DataLoader(test_dataset, batch_size=30, collate_fn=collate, shuffle=False, num_workers=num_workers)

# if config.validation_prop > 0:
#     valid_sampler = SubsetRandomSampler(val_indices)
#     validation_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=valid_sampler, collate_fn=collate)
#     validation_iterator = iter(validation_loader)

n_epochs = config.n_epochs

## Model instanciation and resuming
if args.distributed:
    print('Instanciating DDP')
    model_ddp = nn.parallel.DistributedDataParallel(InteractionSolver(config), device_ids=[args.gpu])
    solver = model_ddp.module
else:
    solver = InteractionSolver(config)
# Save current training random numbers states
# torch_seed = torch.seed()
# np_seed = np.random.get_state()
# # np.random.set_state(np_seed)
# solver.log_dict['torch_seed'] = torch_seed
# solver.log_dict['np_seed'] = np_seed
# solver.log_dict['val_indices'] = val_indices
epoch = 0
iter_counter = 0
saved = 0

if resume:
    epoch = solver.resume(save_dir)

## Training

# Scaler
# scaler = GradScaler()
scaler = None
save_epochs = [config.step_epoch_lr + 5, config.step_epoch_lr + 20, config.step_epoch_lr + 80]
best_score = 999

while (epoch < n_epochs) and (saved > -1):

    if args.distributed:
        train_sampler.set_epoch(epoch)
    
    for batch in iter(train_loader):
        
        ## Dis update
        adv_loss_d = solver.dis_update(batch, scaler=scaler)
    
        ## Gen update
        for n_gen in range(config.n_gen_steps):

            batch = next(gen_iterator, None)
            if batch is None:
                gen_iterator = iter(train_loader)
                batch = gen_iterator.next()

            out = solver.gen_update(batch, writer=writer, scaler=scaler, counter=iter_counter)
            iter_counter += 1
    
    ## Validation & Test
    condition = False
    if args.distributed:
        if is_main_process() and ((epoch % config.save_every == 0) or epoch in save_epochs):
            condition = True
    elif (epoch % config.save_every == 0) or epoch in save_epochs:
        condition = True
    if condition:
        # if config.validation_prop > 0:
        #     with torch.no_grad():
        #         valid_batch = next(validation_iterator, None)
        #         if valid_batch is None:
        #             validation_iterator = iter(validation_loader)
        #             valid_batch = validation_iterator.next()
        #         valid_out = solver.gen_update(valid_batch, train=False)
        save_best = False
        fvds = []
        ffids = []
        tfids = []
        all_w = []
        for i, batch in enumerate(iter(test_loader)):
            if i > 1:
                break
            all_w.append(len(batch[0]))
            with torch.no_grad():
                _, prediction, _, _, _ = solver(batch, seq_len=80, obs_len=3, eval_mode=True)
            fvd, ffid, tfid = solver.metrics(prediction)
            fvds.append(fvd)
            ffids.append(ffid)
            tfids.append(tfid)
        mean_fid = (np.array(ffids) * np.array(all_w)).sum() / np.array(all_w).sum()
        if mean_fid < best_score:
            best_score = mean_fid
            save_best = True
        mean_tfid = (np.array(tfids) * np.array(all_w)).sum() / np.array(all_w).sum()
        mean_fvd = (np.array(fvds) * np.array(all_w)).sum() / np.array(all_w).sum()
        solver.log_dict['metrics']['fid'].append(mean_fid)
        solver.log_dict['metrics']['tfid'].append(mean_tfid)
        solver.log_dict['metrics']['fvd'].append(mean_fvd)
        writer.add_scalar('Metrics/fid', mean_fid, global_step=epoch)
        writer.add_scalar('Metrics/tfid', mean_tfid, global_step=epoch)
        writer.add_scalar('Metrics/fvd', mean_fvd, global_step=epoch)

        saved = solver.save(save_dir, epoch, len(train_loader), new_file=((epoch > 0) & (epoch in save_epochs)), best=save_best)
        if saved == -1:
            print('Early stopping due to gradient overflow')

        args_serialize_path = os.path.join(save_dir, 'args')
        if not os.path.isfile(args_serialize_path):
            # Save the configuration
            with open(args_serialize_path, 'w') as f:
                json.dump(args.__dict__, f)
    if epoch < config.warmup_ep:
        lr_g, lr_d = solver.warmup_lr(epoch)
    else:
        lr_g, lr_d = solver.step_scheduler()

    epoch += 1

# Plot losses in a html file
plot_losses(save_dir, 'training')
# plot_losses(save_dir, 'validation')
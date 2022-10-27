import torch
import torch.nn as nn
from .utils import *


class LinearLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation='none', norm='none', spectral_norm=True, dropout=0):
        super(LinearLayer, self).__init__()

        if spectral_norm:
            self.linear = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels))
        else:
            self.linear = nn.Linear(in_channels, out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = None
            
        if norm == 'in':
            self.norm = nn.InstanceNorm1d(out_channels)
        elif norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels, momentum=0.5)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None
            
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        
    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            if self.norm.__class__.__name__.find('InstanceNorm') > -1:
                x = self.norm(x.unsqueeze(1)).squeeze()
            else:
                x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    
    
class MLP(nn.Module):
    
    def __init__(self, config, input_dim, output_dim, output_activation='tanh', norm=None, depth=None, hidden_dim=None,
                max_dim=1024, spectral_norm=None, twin_out_dim=None):
        super(MLP, self).__init__()
        
        if depth is None:
            depth = config.depth_linear
        if hidden_dim is None:
            hidden_dim = config.base_nodes_linear
        if norm is None:
            norm = config.norm_linear
        if spectral_norm is None:
            spectral_norm = config.spectral_norm
        dropout = config.dropout_linear
        self.twin_out = bool(twin_out_dim)

        model = [LinearLayer(input_dim, hidden_dim, activation='leaky', norm=norm, spectral_norm=spectral_norm, dropout=dropout)]
        
        for i in range(depth):

            if i == depth - 1:
                out_dim = output_dim
                activation = output_activation
                norm = 'none'
                dropout = 0
                
                if twin_out_dim is not None:
                    self.twin_out = nn.ModuleList([
                        LinearLayer(hidden_dim, twin_out_dim, activation=activation, norm=norm, spectral_norm=spectral_norm, dropout=dropout),
                        LinearLayer(hidden_dim, output_dim - twin_out_dim, activation=activation, norm=norm, spectral_norm=spectral_norm, dropout=dropout)
                    ])
                    continue
            else:
                out_dim = min(max_dim, 2 * hidden_dim)
                activation='leaky'
            
            model.append(LinearLayer(hidden_dim, out_dim, activation=activation, norm=norm, spectral_norm=spectral_norm, dropout=dropout))
            hidden_dim = out_dim 

        self.model = nn.Sequential(*model)
        
        self.apply(weight_init)
     
    
    def forward(self, x):

        out = self.model(x)
        if self.twin_out:
            out = torch.cat([twin_out(out).view(out.shape[0], 10, -1) for twin_out in self.twin_out], dim=-1).flatten(start_dim=1)
        
        return out
    
    
    def load_pretrained(self, checkpoint_dir, key):
        
        file = [f for f in os.listdir(checkpoint_dir) if '.pt' in f][0]
        checkpoint = torch.load(os.path.join(checkpoint_dir, file))
        self.load_state_dict(checkpoint[key])


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.n_heads = n_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.final_projection = nn.Linear(input_dim, input_dim)

        self.apply(weight_init)


    def forward(self, q, k, v, pos_token_dim=0, attn_mask=None):
        '''
        The position token is a suffix that does not participate in attention weight computation but is processed with the rest nonetheless
        '''
        in_shape = q.shape
        if len(in_shape) == 4:
            q = q.flatten(start_dim=-2).transpose(1, 2).contiguous()
            k = k.flatten(start_dim=-2).transpose(1, 2).contiguous()
            v = v.flatten(start_dim=-2).transpose(1, 2).contiguous()
        if pos_token_dim > 0:
            encoding = k[..., -pos_token_dim:]
            q = q[..., :-pos_token_dim]
            k = k[..., :-pos_token_dim]
            v = v[..., :-pos_token_dim]
        bs, L, input_dim = q.size()

        queries = self.query(q).view(bs, L, self.n_heads, -1).transpose(1, 2)
        keys = self.key(k).view(bs, L, self.n_heads, -1).transpose(1, 2)
        values = self.value(v).view(bs, L, self.n_heads, -1).transpose(1, 2)
        
        q_k = torch.matmul(queries, keys.transpose(2, 3)) / np.round(np.sqrt(queries.size(-1)), 2)

        if attn_mask is not None:
            attn_mask = attn_mask[None, None, ...]
            q_k = q_k.masked_fill(attn_mask == 1, -1e9)
        
        factors = torch.softmax(q_k, dim=-1)

        context_vect = torch.matmul(factors, values)
        context_vect = self.final_projection(context_vect.transpose(1, 2).flatten(start_dim=-2))
        
        if pos_token_dim > 0:
            context_encoding = torch.matmul(factors, encoding.unsqueeze(1)).mean(dim=1)
            context_vect = torch.cat([context_vect, context_encoding], dim=-1)

        if len(in_shape) == 4:
            context_vect = context_vect.transpose(1, 2).contiguous().view(in_shape)

        return context_vect
    
    
class Pooler(nn.Module):
    
    def __init__(self, config, repeat=True, inhibit_pool=False, same=False, simple_h_switch=False):
        super(Pooler, self).__init__()

        self.repeat = repeat
        self.inhibit_pool = inhibit_pool
        
    def forward(self, h, lengths, dim=0, all_batch=False):

        if self.inhibit_pool:
            return torch.empty(0).cuda()

        indexes = np.cumsum([0] + lengths)
        pooled_h = []

        for i_0, i_f in zip(indexes[:-1], indexes[1:]):
            if dim == 0:
                pooled_h.append(h[i_0:i_f, :].max(dim=0)[0].unsqueeze(0))
            elif dim == 1:
                pooled_h.append(h[:, i_0:i_f, :].max(dim=1)[0].unsqueeze(1))
        pooled_h = torch.cat(pooled_h, dim=dim)
        if self.repeat:
            pooled_h = pooled_h.repeat_interleave(repeats=len(h) if all_batch else torch.Tensor(lengths).long().cuda(), dim=dim)
        
        return pooled_h

    
class DynamicalModel(nn.Module):
    
    def __init__(self, config, hidden_size=None, input_size_norm=256):
        super(DynamicalModel, self).__init__()
        
        self.config = config
        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = config.hidden_size

        if self.config.acceleration_input:
            self.input_size = config.input_size * 3
        elif self.config.velocity_input or (self.config.delta_mod and self.config.autoreg_train):
            self.input_size = config.input_size * 2
        else:
            self.input_size = config.input_size
        self.output_size = config.input_size

        input_size_multiplier = 1 - int(self.config.no_input)

        if self.config.inhibit_pool:
            pool_dim = 0
        else:
            pool_dim = self.hidden_size
        if self.config.init_h_inpt:
            pool_dim += self.hidden_size
            
        # Hidden state is decoded to action space
        normalization = config.norm_deep_out

        # If dynamical model outputs relative displacements, then do activation must allow negative values
        if self.config.velocity_output or self.config.acceleration_output:
            activation = ''
        else:
            activation = 'sigmoid'

        init_mlp_coeff = config.init_mlp_coeff
        if self.config.learn_init_state:
            initial_mlp = [
                nn.Linear(self.input_size, init_mlp_coeff * self.hidden_size),
                nn.ReLU(),
                nn.Linear(init_mlp_coeff * self.hidden_size, 2 * self.hidden_size)
            ]
            if config.dropout_initial_mlp > 0:
                initial_mlp.insert(2, nn.Dropout(config.dropout_initial_mlp))
            self.initial_mlp = nn.Sequential(*initial_mlp)

        ## Deep output net
        self.deep_output = MLP(config, input_dim=self.hidden_size + input_size_multiplier * self.input_size, output_dim=self.output_size, 
            output_activation=activation, norm=normalization, twin_out_dim=20 if config.separate_kp_jac else None)

        # Pooling module
        self.pooler = Pooler(config, inhibit_pool=self.config.inhibit_pool)
        
        # LSTM
        lstm_input_size = input_size_multiplier * self.input_size + pool_dim
        # Additional input to LSTM
        if self.config.init_frame_inpt:
            lstm_input_size += self.input_size
        if self.config.layer_norm:
            self.lstm = LN_LSTM(input_size=lstm_input_size, hidden_size=self.hidden_size)
        else:
            self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.hidden_size)

        # Attention
        if config.attention:
            self.attention_module = SoftAttention(config)
            self.annotation_size = config.annotation_size
        else:
            self.attention_module = None
    

    def initialize_state_vectors(self, batch_size, first_obs=None):

        if self.config.learn_init_state:
            state_tuple = self.initial_mlp(first_obs)
            h_0 = state_tuple[:, :self.hidden_size].contiguous().unsqueeze(0)
            c_0 = state_tuple[:, self.hidden_size:].contiguous().unsqueeze(0)
        elif self.config.zero_init:
            h_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
            c_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        else:
            h_0 = torch.rand(1, batch_size, self.hidden_size).cuda()
            c_0 = torch.rand(1, batch_size, self.hidden_size).cuda()
        
        return (h_0, c_0)

        
    def forward(self, obs_seq, lengths, seq_len, obs_len=None):
        
        bs, obs_seq_len, input_dim = obs_seq.size()
        first_obs = obs_seq[:, 0, :self.input_size]
        x_0 = obs_seq[:, 0, :self.config.input_size]
        state_tuple = self.initialize_state_vectors(bs, first_obs)
        pooled_h = self.pooler(state_tuple[0].squeeze(), lengths) # shape bs, hidden_size

        #####
        ### Encoding of observed sequence
        #####

        obs_outputs = []

        for idx, i in enumerate(range(obs_seq_len)):
            
            x = obs_seq[:, i, :self.output_size].contiguous()
            v = obs_seq[:, i, self.output_size:-self.output_size].contiguous()
            a = obs_seq[:, i, -self.output_size:].contiguous()
            input_data = obs_seq[:, i, :self.input_size].contiguous()

            if self.config.delta_mod:
                delta = x - x_0
                if self.config.autoreg_train:
                    lstm_input = [delta, x_0, pooled_h]
                else:
                    lstm_input = [x_0, pooled_h]
            else:
                if self.config.autoreg_train:
                    lstm_input = [input_data, pooled_h]
                else:
                    lstm_input = [first_obs, pooled_h]
            lstm_input = torch.cat(lstm_input, dim=-1)

            # adding one extra dimension for seq length, which is one
            output, (h_n, c_n) = self.lstm(lstm_input.unsqueeze(0), state_tuple)
            h_n = h_n.squeeze()

            # Deep output
            if self.config.delta_mod:
                if self.config.autoreg_train:
                    deep_out_input = torch.cat([h_n, delta, x_0], dim=-1)
                else:
                    deep_out_input = torch.cat([h_n, x_0], dim=-1)
            else:
                if self.config.autoreg_train:
                    deep_out_input = torch.cat([h_n, input_data], dim=-1)
                else:
                    deep_out_input = torch.cat([h_n, first_obs], dim=-1)
            deep_out = self.deep_output(deep_out_input)

            if self.config.delta_mod:
                last_pos = x_0 + deep_out
                last_vel = last_pos - x
                last_acc = last_vel - v

            elif self.config.acceleration_output:
                # Verlet integration
                last_acc = deep_out
                last_vel = v + 0.5 * (a + last_acc)
                last_pos = x + last_vel + 0.5 * last_acc

            elif self.config.velocity_output:
                last_vel = deep_out
                last_acc = last_vel - v
                last_pos = x + last_vel

            else:
                last_pos = deep_out
                last_vel = deep_out - x
                last_acc = last_vel - v

            out = torch.cat([last_pos, last_vel, last_acc], dim=-1)
            # Pooler
            pooled_h = self.pooler(h_n, lengths) # shape bs, hidden_size

            if i < obs_seq_len - 1:
                obs_outputs.append(out.unsqueeze(1))

            h_n = h_n.unsqueeze(0)
            state_tuple = (h_n, c_n)
        
        if len(obs_outputs) > 0:
            obs_outputs = torch.cat(obs_outputs, dim=1)
        else:
            obs_outputs = torch.empty(0, 0, 0, 0).cuda()

        if seq_len == 0:
            return obs_outputs[:, :obs_len - 1].contiguous(), obs_outputs[:, obs_len - 1:].contiguous()

        #####
        ### Decoding of hidden sequence
        #####
        len_to_decode = seq_len - 1
        outputs = [out.unsqueeze(1)]
        last_observed_h = state_tuple[0].squeeze().clone()

        # Mb add noise
        if self.config.noise_prop > 0:
            pooled_h += generate_noise(pooled_h, lengths, self.config.noise_prop)

        for i in range(len_to_decode):
            
            if self.config.no_input:
                input_data = torch.Tensor([]).cuda()
            elif self.config.acceleration_input:
                input_data = torch.cat([last_pos, last_vel, last_acc], dim=-1)
            elif self.config.velocity_input:
                input_data = torch.cat([last_pos, last_vel], dim=-1)
            else:
                input_data = last_pos
            
            if self.config.delta_mod:
                delta = last_pos - x_0
                if self.config.autoreg_train:
                    lstm_input = [delta, x_0, pooled_h]
                else:
                    lstm_input = [x_0, pooled_h]
            else:
                if self.config.autoreg_train:
                    lstm_input = [input_data, pooled_h]
                else:
                    lstm_input = [first_obs, pooled_h]
            lstm_input = torch.cat(lstm_input, dim=-1)

            # adding one extra dimension for seq length, which is one
            output, (h_n, c_n) = self.lstm(lstm_input.unsqueeze(0), state_tuple)
            h_n = h_n.squeeze()

            if self.config.delta_mod:
                if self.config.autoreg_train:
                    deep_out_input = torch.cat([h_n, delta, x_0], dim=-1)
                else:
                    deep_out_input = torch.cat([h_n, x_0], dim=-1)
            else:
                if self.config.autoreg_train:
                    deep_out_input = torch.cat([h_n, input_data], dim=-1)
                else:
                    deep_out_input = torch.cat([h_n, first_obs], dim=-1)
            deep_out = self.deep_output(deep_out_input)

            if self.config.delta_mod:
                last_acc = x_0 + deep_out - last_pos - last_vel
                last_vel = x_0 + deep_out - last_pos
                last_pos = x_0 + deep_out
            
            elif self.config.acceleration_output:
                # Verlet integration
                last_vel += 0.5 * (last_acc + deep_out)
                last_acc = deep_out
                last_pos += last_vel + 0.5 * last_acc

            elif self.config.velocity_output:
                last_acc = deep_out - last_vel
                last_vel = deep_out
                last_pos += last_vel

            else:
                last_acc = (deep_out - last_pos) - last_vel
                last_vel = deep_out - last_pos
                last_pos = deep_out

            out = torch.cat([last_pos, last_vel, last_acc], dim=-1)
            outputs.append(out.unsqueeze(1))

            # Pooler
            pooled_h = self.pooler(h_n, lengths) # shape bs, hidden_size

            if self.attention_module is not None:
                pooled_h += self.attention_module(annotations, h_n)
                annotations = torch.cat([annotations, h_n.unsqueeze(1)], dim=1)

            h_n = h_n.unsqueeze(0)
            state_tuple = (h_n, c_n)

        outputs = torch.cat(outputs, dim=1)
        
        return obs_outputs, outputs
    
    
class Discriminator(nn.Module):
    
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        self.config = config
        self.gan_loss = config.gan_loss
        self.lambda_gp = config.gp_weight
        self.lambda_pooling = config.pooling_weight
        self.lambda_seq_stream = config.seq_stream_weight
        self.lambda_landmark = config.landmark_realism_weight
        self.cnn_dis = config.cnn_dis
        self.dual_stream = config.dual_stream
        self.double_seq_stream = config.double_seq_stream

        if config.keypoints:
            # If unsupervised keypoints, inputs consist of all three coordinates orders, plus jacobian orders 1 and 2 (second order jacobian dynamics are discarded)
            self.input_size = int(config.input_size * (2 + 1 / 3))
        elif config.acceleration:
            self.input_size = config.input_size * 3
            input_size_list = [config.input_size] * 3
        elif config.velocity:
            self.input_size = config.input_size * 2
            input_size_list = [config.input_size] * 2
        else:
            self.input_size = config.input_size
            input_size_list = [config.input_size]

        # Pooler
        self.pooler = Pooler(config, repeat=False)

        # Individual landmarks realism score
        self.ldk_classif_inpt_size = self.input_size if config.keypoints else 3 * config.input_size # self.input_size
        self.landmark_classifier = MLP(config, input_dim=self.ldk_classif_inpt_size, output_dim=1, output_activation='none')

        if self.gan_loss not in ['hinge', 'gp']:
            self.sigmoid_out = nn.Sigmoid()
            
        # Subsequence length for individual and sequence streams
        self.windows = config.windows[str(config.cnn_config)]
        self.interaction_windows = config.windows[str(config.interaction_config)]


        if self.config.net_type_D == 'transformer':
            if config.proj_score_y:
                encoder_out_dim = config.emb_dim_D
                self.projection_disc = ProjectionDisc(encoder_out_dim)
                if self.dual_stream:
                    self.interaction_projection_disc = ProjectionDisc(encoder_out_dim)
            else:
                encoder_out_dim = 1

            self.encoder = TransformerDecoder(input_size_list, encoder_out_dim, config.n_heads_D, config.n_blocks_D, config.hidden_size_D, 
                residual=config.residual, pos_enc=config.pos_encoding, cls_tk=True, dropout=config.dropout_linear, temp_mixing=config.temp_mixing,
                emb_dim=config.emb_dim_D)
            if self.dual_stream:
                if self.double_seq_stream:
                    pool_type = None
                else:
                    pool_type=config.pool_type
                self.interaction_encoder = TransformerDecoder(input_size_list, encoder_out_dim, config.n_heads_D, config.n_blocks_D, config.hidden_size_D, 
                residual=config.residual, pos_enc=config.pos_encoding, cls_tk=True, batch_xa=False, dropout=config.dropout_linear, temp_mixing=config.temp_mixing,
                emb_dim=config.emb_dim_D, pool_type=pool_type)
        
        elif self.config.net_type_D == 'cnn':
            self.encoder = ConvDis(self.config, self.input_size, False)
            if self.dual_stream:
                self.interaction_encoder = ConvDis(self.config, self.input_size, True)

        else:
            h_dim = config.hidden_size_D

            # LSTMs
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=h_dim)
            if self.dual_stream:
                self.interaction_lstm = nn.LSTM(input_size=self.input_size, hidden_size=h_dim)

            # Projection discriminators
            self.projection_disc = ProjectionDisc(h_dim, net_type=config.proj_dis_type)
            if self.dual_stream:
                self.interaction_projection_disc = ProjectionDisc(h_dim, net_type=config.proj_dis_type)
        
        
    def forward(self, conditioning, out_sequence, lengths):

        ### Suppression of starting spatial position, origin is set at a nose landmark (number 27)
        if self.config.random_translate_dis_input and (len(conditioning.shape) != 0):
            randn = (torch.rand(conditioning.shape[0], 2) * 0.6 - 0.3).cuda()
            starting_position = randn.unsqueeze(1).repeat(1, 1, 68)
            zeros = torch.zeros_like(starting_position)
            starting_position = torch.cat([starting_position, zeros, zeros], dim=-1)
            conditioning = conditioning - starting_position            
            out_sequence = out_sequence - starting_position
        ###

        ### Suppression of starting y axis orientation if dim = 3
        if (self.config.data_dim == 3) and (self.config.normalize_head_orientation):
            if len(conditioning.shape) != 0:
                conditioning = b_rotate_3D_tensor(conditioning)
            out_sequence = b_rotate_3D_tensor(out_sequence)
        ###

        ####
        ### Frame discriminator
        ####
        ldk_classif_input = out_sequence[..., :self.ldk_classif_inpt_size].contiguous()
        # If 3D, rotate the sequence in a random direction and assess the individual realism of rotated tensors
        if (self.config.data_dim == 3) and (self.config.random_rotate_dis_input):
            max_angle = np.pi / 3
            bs = ldk_classif_input.shape[0]
            theta_y = (1.5 * (2 * torch.rand(bs) - 1) * max_angle).clamp(min=-max_angle, max=max_angle)
            sin_half = torch.sin(theta_y / 2)
            cos_half = torch.cos(theta_y / 2)
            ldk_classif_input = b_rotate_3D_tensor(ldk_classif_input, sin_half, cos_half)
        
        individual_realism = self.landmark_classifier(ldk_classif_input.flatten(end_dim=1))

        ####
        ### Sequential discriminator
        ####
        positions = out_sequence[..., :self.config.input_size].contiguous()
        initial_frame = positions[:, 0, :].contiguous()
        out_sequence = out_sequence[..., :self.input_size].contiguous()

        seq_len = out_sequence.size(1)
        conditioning = conditioning[..., :self.input_size].contiguous()
        obs_len = conditioning.size(1)
        if obs_len > 0:
            sequence = torch.cat([conditioning, out_sequence], dim=1)
        else:
            sequence = out_sequence

        # Sample chunk widths if needed
        self.sample_chunk_width()

        if self.config.net_type_D == 'transformer':

            if self.lambda_seq_stream > 0:
                local_scores = torch.zeros(sequence.shape[0], 1).cuda()
                y = self.encoder(conditioning)
                for idx, window in enumerate(self.windows):
                    # Start indices
                    stride = int(window / 2)
                    local_idx = np.arange(max(-obs_len, stride - window), seq_len - window + 1, stride)
                    # Slice chunks and stack them
                    chunks = torch.cat([sequence[:, obs_len + i:obs_len + i + window, :] for i in local_idx], dim=0)
                    ##
                    if hasattr(self, 'projection_disc'):
                        proj_dis_inpt = self.encoder(chunks)
                        local_scores += self.projection_disc(proj_dis_inpt, y, local_idx).mean(dim=0) 
                    else:
                        local_scores += self.encoder(chunks).view(len(local_idx), -1, 1).mean(dim=0)                
                output = local_scores / (idx + 1)
            else:
                output = torch.tensor(0.0).cuda()
                
            if self.double_seq_stream:
                local_scores_inter = torch.zeros(sequence.shape[0], 1).cuda()
            else:
                local_scores_inter = torch.zeros(int(sequence.shape[0] / 2), 1).cuda()
            if self.dual_stream:
                y = self.interaction_encoder(conditioning)
                for idx, window in enumerate(self.interaction_windows):
                    stride = int(window / 2)
                    local_idx = np.arange(max(-obs_len, stride - window), seq_len - window + 1, stride)
                    chunks = torch.cat([sequence[:, obs_len + i:obs_len + i + window, :] for i in local_idx], dim=0)
                    if hasattr(self, 'interaction_projection_disc'):
                        proj_dis_inpt = self.interaction_encoder(chunks)
                        local_scores_inter += self.interaction_projection_disc(proj_dis_inpt, y, local_idx).mean(dim=0) 
                    else:
                        local_scores_inter += self.interaction_encoder(chunks).view(len(local_idx), -1, 1).mean(dim=0)
                interaction_out = local_scores_inter / (idx + 1)
            else:
                interaction_out = torch.tensor(0.0).cuda()

        elif self.config.net_type_D == 'cnn':
            output = self.encoder(sequence)
            if self.dual_stream:
                interaction_out = self.interaction_encoder(sequence)
            else:
                interaction_out = torch.tensor(0.0).cuda()

        else:
            if self.lambda_seq_stream > 0:
                output = self.forward_local_dis(self.lstm, self.projection_disc, conditioning, sequence, obs_len, seq_len, initial_frame)
            else:
                output = torch.tensor(0.0).cuda()
            # Interaction stream
            if self.dual_stream:
                if self.double_seq_stream:
                    pool = False
                else:
                    pool = True
                interaction_out = self.forward_local_dis(self.interaction_lstm, self.interaction_projection_disc, conditioning, sequence, obs_len, seq_len, 
                initial_frame, lengths, pool)
            else:
                interaction_out = torch.tensor(0.0).cuda()

        if self.gan_loss not in ['hinge', 'gp']:
            output = self.sigmoid_out(output)
            interaction_out = self.sigmoid_out(interaction_out)
            individual_realism = self.sigmoid_out(individual_realism)
        
        return output, interaction_out, individual_realism


    def forward_local_dis(self, lstm, projection_disc, conditioning, sequence, obs_len, seq_len, initial_frame=None, lengths=None, interaction=False):
        ## Local projection disciminator, y stands for the last conditioning hidden vector from which to compute dot product in projection dis  
        bs, total_len, input_size = sequence.size()

        # Computation of the conditioning vector
        all_y, (y, _) = lstm(conditioning.transpose(0, 1))
        y = y.squeeze()

        # Initialization of scores / windows differently for individual & interaction streams
        if interaction:
            local_scores = torch.zeros(len(lengths), 1).cuda()
            windows = self.interaction_windows
            y = self.pooler(y, lengths)
        else:
            local_scores = torch.zeros(bs, 1).cuda()
            windows = self.windows

        for idx, window in enumerate(windows):
            stride = int(window / 2)

            # Subsequences start indices
            # local_idx = np.arange(0, obs_len + seq_len - window + 1, stride)
            # overlapping_chunks = torch.cat([sequence[:, i:i + window, :] for i in local_idx], dim=0)
            local_idx = np.arange(max(-obs_len, stride - window), seq_len - window + 1, stride)
            overlapping_chunks = torch.cat([sequence[:, obs_len + i:obs_len + i + window, :] for i in local_idx], dim=0)
            _, (local_out, c) = lstm(overlapping_chunks.transpose(0, 1))
            proj_dis_input = local_out.squeeze()
            # proj_dis_bs = bs
            if interaction:
                local_pooled_vect = self.pooler(local_out.squeeze().view(len(local_idx), bs, -1), lengths, dim=1)
                proj_dis_input = local_pooled_vect.flatten(end_dim=1)
                # proj_dis_bs = 1 if self.config.batch_pool else len(lengths)

            local_scores += projection_disc(proj_dis_input, y, distances=local_idx).mean(dim=0)

        return local_scores / (idx + 1)

    
    def sample_chunk_width(self):
        if self.config.cnn_config == 100:
            random_sizes = sample_exponential(np.random.uniform(size=3), min_value=3, max_value=self.config.seq_len)
            setattr(self, 'windows', np.unique([int(s) for s in random_sizes]))
            print(f'Chunk window size: {self.windows}')
        if self.config.interaction_config == 100:
            inter_random_sizes = sample_exponential(np.random.uniform(size=3), min_value=3, max_value=self.config.seq_len)
            setattr(self, 'interaction_windows', np.unique([int(s) for s in inter_random_sizes]))
    
    
    def compute_gen_loss(self, prediction, ground_truth, conditioning, lengths, eps=1e-8):
        
        fake_out, fake_interaction_out, fake_individual_realism = self.forward(conditioning, prediction, lengths)
        
        if self.gan_loss in ['hinge', 'gp']:
            adv_loss_individual = -torch.mean(fake_out)
            adv_loss_interaction = -torch.mean(fake_interaction_out)
            adv_loss_landmarks = -torch.mean(fake_individual_realism)
        else:
            adv_loss_individual = -torch.mean(torch.log(torch.clamp(fake_out, eps, 1 - eps)))
            adv_loss_interaction = -torch.mean(torch.log(torch.clamp(fake_interaction_out, eps, 1 - eps)))
            adv_loss_landmarks = -torch.mean(torch.log(torch.clamp(fake_individual_realism, eps, 1 - eps)))

        out = self.lambda_seq_stream * adv_loss_individual + self.lambda_pooling * adv_loss_interaction + self.lambda_landmark * adv_loss_landmarks
        
        return (out, adv_loss_individual, adv_loss_interaction, adv_loss_landmarks)
        
        
    def compute_dis_loss(self, prediction, ground_truth, conditioning, lengths, eps=1e-8):        
        
        batch_size = ground_truth.size(0)
        
        fake_out, fake_interaction_out, fake_individual_realism = self.forward(conditioning, prediction, lengths)
        all_f_out = (fake_out.mean().item(), fake_interaction_out.mean().item(), fake_individual_realism.mean().item())
        real_out, real_interaction_out, real_individual_realism = self.forward(conditioning, ground_truth, lengths)
        all_r_out = (real_out.mean().item(), real_interaction_out.mean().item(), real_individual_realism.mean().item())
        
        if self.gan_loss == 'hinge':
            adv_loss_individual = torch.mean(torch.max(torch.zeros_like(fake_out), 1 + fake_out)) + \
                            torch.mean(torch.max(torch.zeros_like(real_out), 1 - real_out))
            adv_loss_interaction = torch.mean(torch.max(torch.zeros_like(fake_interaction_out), 1 + fake_interaction_out)) + \
                            torch.mean(torch.max(torch.zeros_like(real_interaction_out), 1 - real_interaction_out))
            adv_loss_landmarks = torch.mean(torch.max(torch.zeros_like(fake_individual_realism), 1 + fake_individual_realism)) + \
                            torch.mean(torch.max(torch.zeros_like(real_individual_realism), 1 - real_individual_realism))
        
        elif self.gan_loss == 'gp':
            with torch.backends.cudnn.flags(enabled=False):
                # Calculation of gradient penalty
                epsilon = torch.rand(batch_size, 1).repeat(1, int(ground_truth.numel() / batch_size)).view(ground_truth.size()).cuda()
                interpolates = epsilon * prediction.detach() + (1 - epsilon) * ground_truth.detach()
                interpolates.requires_grad_(True)

                interpolates_out, interpolates_interaction_out, interpolates_individual_realism = self.forward(conditioning, interpolates, lengths)

                # Individual sequences
                grad_penalty = gradient_penalty(inputs=interpolates, outputs=interpolates_out)
                adv_loss_individual = torch.mean(fake_out) - torch.mean(real_out) + self.lambda_gp * grad_penalty

                # Interactions
                if len(interpolates_interaction_out.shape) > 0:
                    grad_penalty = gradient_penalty(inputs=interpolates, outputs=interpolates_interaction_out)
                    adv_loss_interaction = torch.mean(fake_interaction_out) - torch.mean(real_interaction_out) + self.lambda_gp * grad_penalty
                else:
                    adv_loss_interaction = torch.tensor(0.0).cuda()

                # Landmarks
                grad_penalty = gradient_penalty(inputs=interpolates, outputs=interpolates_individual_realism)
                adv_loss_landmarks = torch.mean(fake_individual_realism) - torch.mean(real_individual_realism) + self.lambda_gp * grad_penalty
            
        else: # original GAN loss            
            adv_loss_individual = -(torch.mean(torch.log(torch.clamp(real_out, eps, 1 - eps))) + \
                           torch.mean(torch.log(torch.clamp(1 - fake_out, eps, 1 - eps))))
            adv_loss_interaction = -(torch.mean(torch.log(torch.clamp(real_interaction_out, eps, 1 - eps))) + \
                           torch.mean(torch.log(torch.clamp(1 - fake_interaction_out, eps, 1 - eps))))
            adv_loss_landmarks = -(torch.mean(torch.log(torch.clamp(real_individual_realism, eps, 1 - eps))) + \
                           torch.mean(torch.log(torch.clamp(1 - fake_individual_realism, eps, 1 - eps))))


        out = self.lambda_seq_stream * adv_loss_individual + self.lambda_pooling * adv_loss_interaction + self.lambda_landmark * adv_loss_landmarks
        
        return (out, adv_loss_individual, adv_loss_interaction, adv_loss_landmarks, all_f_out, all_r_out)



class ProjectionDisc(nn.Module):
    
    def __init__(self, input_dim, net_type='default'):
        super(ProjectionDisc, self).__init__()
        
        if net_type == 'light':
            dimension = 1024
        else:
            dimension = 2 * input_dim

        if 'mlp' in net_type:
            self.phi = nn.Sequential(*[
                nn.Linear(input_dim, dimension),
                nn.LayerNorm(dimension),
                nn.LeakyReLU(),
                nn.Linear(dimension, dimension),
                nn.LayerNorm(dimension),
                nn.LeakyReLU(),
                nn.Linear(dimension, dimension)
            ])
        else:
            self.phi = LinearLayer(input_dim, dimension, 'relu')
        self.psi = LinearLayer(dimension, dimension, 'relu')
        # self.psi = LinearLayer(dimension, 1)
        self.V = LinearLayer(dimension, input_dim)
        self.A = LinearLayer(dimension, 1)
        self.beta = nn.Parameter(torch.ones(1).mul_(5), requires_grad=True)
        
    def forward(self, x, y, distances):
        # x dim: len(distances) * bs, dim
        # y dim: bs, dim
        batch_size = y.shape[0]

        distances = (torch.tensor(distances).clamp(1)).float().cuda()
        
        phi = self.phi(x)

        # return self.A(self.psi(phi)).view(len(distances), batch_size, 1)

        y = y.repeat(len(distances), 1)
        # y = torch.ones_like(y)
        # y = y / y.norm(dim=-1, keepdim=True)

        V_out = self.V(phi)
        # V_out = V_out / V_out.norm(dim=-1, keepdim=True)
        
        projection = torch.matmul(y.unsqueeze(1), V_out.unsqueeze(2)).squeeze(1)
        coef = distances ** (-1 / self.beta)
        # coef = torch.ones_like(coef)
        coef = coef.repeat_interleave(batch_size)
        
        #return (coef.unsqueeze(1) * projection + self.psi(phi)).view(len(distances), batch_size, 1)
        return self.A(coef.unsqueeze(1) * projection + self.psi(phi)).view(len(distances), batch_size, 1)


### Modules for a CNN discriminator

class ConvDis(nn.Module):
    def __init__(self, config, dim, dual):
        super(ConvDis, self).__init__()
        self.blocks = nn.ModuleList(
        [
            nn.ModuleList([nn.Sequential(*[IRBTempMixer(dim, 3) for _ in range(config.cnn_block_depth)]), nn.MaxPool1d(2)]) \
                for _ in range(config.cnn_depth)
        ])
        
        if dual:
            self.dual_mixer = nn.Sequential(*[nn.Linear(dim, dim), nn.LeakyReLU(0.1)])
    
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(dim, 1)
        
    def forward(self, x):
        bs, length, dim = x.shape
        for block, maxpool in self.blocks:
            x = block(x)
            x = maxpool(x.transpose(1, 2)).transpose(1, 2)
        if hasattr(self, 'duak_mixer'):
            x = self.dual_mixer(x.view(int(bs / 2), 2, -1, dim).max(dim=1)[0])
        out = self.classifier(self.final_pool(x.transpose(1, 2)).squeeze())
        return out

### Transformer block / plain simple Transformer decoder

class TransformerBlock(nn.Module):

    def __init__(self, input_dim, batch_xa=False, residual=True, n_heads=3, proj_dim=512, proj_dim_pos_token=256, dropout=0):
        super(TransformerBlock, self).__init__()
        
        # self.pos_tk_dim = position_token_dim
        # self.custom_sa = custom_sa
        self.residual = residual
        self.batch_xa = batch_xa
        
        # if self.custom_sa:
        #     self.mh_attention = MultiHeadAttention(input_dim, n_heads)
        #     self.dropout_mha = nn.Dropout(dropout)
        #     if self.batch_xa:
        #         self.batch_x_attention = MultiHeadAttention(input_dim, n_heads)
        #         self.norm_bxa = nn.LayerNorm(input_dim)
        #         self.dropout_bxa = nn.Dropout(dropout)
        # else:
        self.mh_attention = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.dropout_mha = nn.Dropout(dropout)
        if self.batch_xa:
            self.batch_x_attention = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
            self.norm_bxa = nn.LayerNorm(input_dim)
            self.dropout_bxa = nn.Dropout(dropout)
        self.norm_mha = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(*[
            nn.Linear(input_dim, proj_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, input_dim),
            nn.Dropout(dropout)
        ])
        self.norm_ff = nn.LayerNorm(input_dim)
        
        # ####
        # ### Custom MHA with an appended position token
        # ####
        # if self.custom_sa and (self.pos_tk_dim > 0):
        #     self.norm_mha_pos_tk = nn.LayerNorm(self.pos_tk_dim)
        #     if self.batch_xa:
        #         self.norm_bxa_pos_tk = nn.LayerNorm(self.pos_tk_dim)
        #     self.ff_pos_tk = nn.Sequential(*[
        #         nn.Linear(self.pos_tk_dim, proj_dim_pos_token),
        #         nn.LeakyReLU(0.2),
        #         nn.Dropout(dropout),
        #         nn.Linear(proj_dim_pos_token, self.pos_tk_dim),
        #         nn.Dropout(dropout)
        #     ])
        #     self.norm_ff_pos_tk = nn.LayerNorm(self.pos_tk_dim)
        
    def forward(self, seq, attn_mask=None, pool=False):

        b, t, d = seq.size()
        
        # ####
        # ### Custom MHA
        # ####
        # if self.custom_sa:
        #     sa_out = seq + self.dropout_mha(self.mh_attention(seq, seq, seq, pos_token_dim=self.pos_tk_dim, attn_mask=attn_mask)) # Size: bs, length, dim + position_encoding_dim
            
        #     ### Position token
        #     if self.pos_tk_dim > 0:
        #         sa_out_pos_tk = self.norm_mha_pos_tk(sa_out[..., -self.pos_tk_dim:])
        #         sa_out = self.norm_mha(sa_out[..., :-self.pos_tk_dim])
        #         if self.batch_xa:
        #             sa_out = torch.cat([sa_out, sa_out_pos_tk], dim=-1)
        #             keys = sa_out.view(-1, 2, t, d)[:, [1, 0]].clone()
        #             keys = keys.flatten(end_dim=1)
        #             sa_out = sa_out + self.dropout_bxa(self.batch_x_attention(sa_out, keys, keys, pos_token_dim=self.pos_tk_dim, attn_mask=attn_mask))
        #             sa_out_pos_tk = self.norm_bxa_pos_tk(sa_out[..., -self.pos_tk_dim:])
        #             sa_out = self.norm_bxa(sa_out[..., :-self.pos_tk_dim])
        #         return torch.cat([
        #             self.norm_ff(sa_out + self.ff(sa_out)),
        #             self.norm_ff_pos_tk(sa_out_pos_tk + self.ff_pos_tk(sa_out_pos_tk))
        #         ], dim=-1)

        #     else:
        #         sa_out = self.norm_mha(sa_out)
        #         if self.batch_xa:
        #             keys = sa_out.view(-1, 2, t, d)[:, [1, 0]].clone()
        #             keys = keys.flatten(end_dim=1)
        #             sa_out = self.norm_bxa(sa_out + self.dropout_bxa(self.batch_x_attention(sa_out, keys, keys, pos_token_dim=self.pos_tk_dim, attn_mask=attn_mask)))
        #         return self.norm_ff(sa_out + self.ff(sa_out))
        #### 
        ### Regular MHA
        ####
        # else:
        sa_out = self.norm_mha(seq + self.dropout_mha(self.mh_attention(seq, seq, seq, need_weights=False, attn_mask=attn_mask)[0]))
        if self.batch_xa:
            keys = sa_out.view(-1, 2, t, d)[:, [1, 0]].clone()
            keys = keys.flatten(end_dim=1)
            sa_out = self.norm_bxa(sa_out + self.dropout_bxa(self.batch_x_attention(sa_out, keys, keys, need_weights=False, attn_mask=attn_mask)[0]))
        if pool:
            sa_out = sa_out.view(int(b / 2), 2, t, d).max(dim=1)[0]
        if self.residual:
            return self.norm_ff(sa_out + self.ff(sa_out))
        else:
            return self.norm_ff(self.ff(sa_out))


class TransformerDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, n_heads, n_blocks, proj_dim, residual=True, pos_enc=False, prepend_inpt=False, cls_tk=False, batch_xa=False,
     dropout=0, temp_mixing=False, emb_dim=None, pool_type=None, n_position=200):
        super(TransformerDecoder, self).__init__()

        self.dim = np.cumsum([0] + in_dim)
        total_in_dim = self.dim[-1]
        # self.custom_sa = custom_sa
        # self.pos_tk_dim = position_token_dim
        self.pos_enc = pos_enc
        self.prepend_inpt = prepend_inpt
        self.cls_tk = cls_tk
        self.n_position = n_position
        if emb_dim is None:
            emb_dim = total_in_dim

        # if custom_sa and (position_token_dim > 0):
        #     pos_enc_dim = position_token_dim
        #     decoder_out_dim += position_token_dim
        # else:
        #     pos_enc_dim = emb_dim

        self.pool_type = pool_type
        if pool_type == 'xa':
            batch_xa = True
        elif pool_type == 'cat':
            total_in_dim *= 2
            emb_dim *= 2
        pos_enc_dim = emb_dim
        decoder_out_dim = emb_dim
        if prepend_inpt:
            decoder_out_dim += emb_dim

        self.embedding = nn.Linear(total_in_dim, emb_dim)
        # self.embedding = nn.ModuleList(
        #     [nn.Linear(dim, dim) for dim in in_dim]
        # )

        if temp_mixing:
            self.temp_mixer = IRBTempMixer(emb_dim, 3)

        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, batch_xa, residual, n_heads=n_heads,
         proj_dim=proj_dim, dropout=dropout) for _ in range(n_blocks)])
        
        # self.linear_out = nn.Linear(decoder_out_dim, out_dim)

        self.deep_out = nn.Sequential(*[
            nn.Linear(decoder_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_dim, out_dim)
        ])        

        self.register_buffer('pos_table', encode_position(pos_enc_dim, n_position).unsqueeze(0).cuda())
        self.dropout_posenc = nn.Dropout(dropout)

        if self.cls_tk:
            self.cls_token = nn.Parameter(torch.randn(emb_dim).mul_(0.1), requires_grad=True)
            # self.register_parameter('cls_token', nn.Parameter(torch.randn(self.dim).mul_(0.1), requires_grad=True))


    def forward(self, seq, mask=False, return_all_seq=False):
        
        _, length, dim = seq.shape
        if mask:
            # forward_mask = (torch.triu(torch.ones(length, length), diagonal=1) + torch.tril(torch.ones(length, length), diagonal=-35))\
            #     .bool().cuda()
            forward_mask = torch.triu(torch.ones(length, length), diagonal=1).bool().cuda()

        else:
            forward_mask = None

        #####
        ## Embedding
        #####
        if self.pool_type == 'cat':
            seq = seq.view(-1, 2, length, dim)
            seq = torch.cat([seq[:, 0], seq[:, 1]], dim=-1)
        block_out = self.embedding(seq)
        if self.pool_type == 'early_pool':
            block_out = block_out.view(-1, 2, length, block_out.shape[-1]).max(dim=1)[0]
        # block_out = torch.cat([embedding(seq[..., self.dim[i]:self.dim[i + 1]].contiguous()) for i, embedding in enumerate(self.embedding)], dim=-1)
        
        #####
        ## Positional encoding
        #####
        # Add class token
        if self.cls_tk:
            block_out = torch.cat([self.cls_token[None, None, :].repeat(block_out.size(0), 1, 1), block_out], dim=1)
            length += 1
        # Scale input
        scale_fact = 0
        if self.pos_enc:
            scale_fact = 0.02
            # block_out = block_out * (self.dim[-1] ** 0.5)
        # Add position encoding
        position_encoding = scale_fact * self.pos_table[:, :length].clone().detach()
        # if self.custom_sa and (self.pos_tk_dim > 0):
        #     block_out = torch.cat([block_out, position_encoding.repeat(block_out.size(0), 1, 1)], dim=-1)
        # else:
        block_out = block_out + position_encoding

        #####
        ## CNN temporal mixing
        #####
        if hasattr(self, 'temp_mixer'):
            block_out = self.temp_mixer(block_out)

        block_out = self.dropout_posenc(block_out)

        #####
        ## Process through transformer layers
        #####
        for i, trans_block in enumerate(self.blocks):
            block_out = trans_block(block_out, attn_mask=forward_mask, pool=(self.pool_type == 'late_pool') and (i == 0))

        if self.cls_tk:
            return self.deep_out(block_out[:, 0, :])

        # Skip connection
        if self.prepend_inpt:
            block_out = torch.cat([seq, block_out], dim=-1)

        if return_all_seq:
            output = self.deep_out(block_out)
        else:
            output = self.deep_out(block_out[:, -1, :])

        return output


class TransformerGenerator(nn.Module):

    def __init__(self, config):
        super(TransformerGenerator, self).__init__()

        self.config = config
        self.dim = [config.input_size]
        if config.velocity_input:
            self.dim.append(config.input_size)
        if config.acceleration_input:
            self.dim.append(config.input_size)
        self.mask = config.forward_mask 

        self.output_size = config.input_size
        self.decoder = TransformerDecoder(self.dim, self.output_size, config.n_heads, config.n_blocks, config.hidden_size, 
            prepend_inpt=config.pool_in_do, batch_xa=config.dual_stream and not config.inhibit_pool, dropout=config.dropout_linear, 
            temp_mixing=config.temp_mixing, emb_dim=config.emb_dim)

    def forward(self, inpt_seq, obs_len, len_to_decode):      
        
        x = inpt_seq[..., :self.output_size].contiguous()
        v = inpt_seq[..., self.output_size:-self.output_size].contiguous()
        a = inpt_seq[..., -self.output_size:].contiguous()
        input_data = x
        if self.config.velocity_input:
            input_data = torch.cat([input_data, v], dim=-1)
        if self.config.acceleration_input:
            input_data = torch.cat([input_data, a], dim=-1)
        
        # Decoder output is a instantaneous velocity
        vel = self.decoder(input_data, mask=True, return_all_seq=True)
        acc = vel - v
        pos = x + vel

        out = torch.cat([pos, vel, acc], dim=-1)

        if len_to_decode == 0:
            return out[:, :obs_len - 1].contiguous(), out[:, obs_len - 1:-1].contiguous()

        if out.size(1) > 1:
            obs_outputs = out[:, :-1, :].contiguous()
        else:
            obs_outputs = torch.empty(0, 0, 0).cuda()
        out = out[:, -1, :]

        #####
        ### Decoding
        #####
        # gen_seq = torch.cat([inpt_seq[:, 0, :].unsqueeze(1), obs_outputs, out.unsqueeze(1)], dim=1)
        gen_seq = torch.cat([inpt_seq, out.unsqueeze(1)], dim=1)
        last_vel = vel[:, -1, :]
        last_pos = pos[:, -1, :]

        for i in range(len_to_decode - 1):
            
            input_data = gen_seq[..., :self.output_size].contiguous()
            if self.config.velocity_input:
                v = gen_seq[..., self.output_size:-self.output_size].contiguous()
                input_data = torch.cat([input_data, v], dim=-1)
            if self.config.acceleration_input:
                a = gen_seq[..., -self.output_size:].contiguous()
                input_data = torch.cat([input_data, a], dim=-1)

            decoder_out = self.decoder(input_data, mask=self.mask)

            # Decoder output is a instantaneous velocity
            last_acc = decoder_out - last_vel
            last_vel = decoder_out
            last_pos += last_vel

            out = torch.cat([last_pos, last_vel, last_acc], dim=-1)
            gen_seq = torch.cat([gen_seq, out.unsqueeze(1)], dim=1)

        outputs = gen_seq[:, inpt_seq.size(1):, :].contiguous()
        
        return obs_outputs, outputs


def encode_position(dim, length):

    encoding = torch.matmul(torch.arange(length, dtype=torch.float32).unsqueeze(1),
                1 / (1e3 ** (torch.arange(0, dim, step=2) / dim)).unsqueeze(0))

    sin = torch.sin(encoding)
    cos = torch.cos(encoding)
    encoding = torch.stack([sin, cos], dim=-1).flatten(start_dim=-2)

    return encoding

class SRNNGenerator(nn.Module):

    def __init__(self, config):
        super(SRNNGenerator, self).__init__()

        self.config = config
        self.dim = config.input_size
        if config.velocity_input:
            self.dim += config.input_size
        if config.acceleration_input:
            self.dim += config.input_size

        self.output_size = config.input_size

        ## Subnetworks instantiation

        hidden_size = self.config.hidden_size
        if self.config.layer_norm:
            lstm = LN_LSTM(input_size=self.dim, hidden_size=hidden_size)
        else:
            lstm = nn.LSTM(input_size=self.dim, hidden_size=hidden_size)

        # Posterior inference
        self.backward_lstm = nn.LSTM(input_size=self.dim + hidden_size, hidden_size=hidden_size)
        self.posterior = MLP(config, input_dim=2 * hidden_size, output_dim=2 * hidden_size, output_activation='')

        # Initial latent variable
        self.init_z_layer = nn.Linear(self.dim, hidden_size)

        # Prior and conditional MLPs
        self.prior = MLP(config, input_dim=2 * hidden_size, output_dim=2 * hidden_size, output_activation='')
        self.conditional = MLP(config, input_dim=2 * hidden_size, output_dim=2 * self.output_size, output_activation='')

        if self.config.learn_init_state:
            self.zero_state_initializer = nn.Linear(self.output_size, 2 * self.hidden_size)

        
    def initialize_state_vectors(self, batch_size, first_obs=None):

        if self.config.learn_init_state:
            state_tuple = self.zero_state_initializer(first_obs)
            h_0 = state_tuple[:, :self.hidden_size].contiguous().unsqueeze(0)
            c_0 = state_tuple[:, self.hidden_size:].contiguous().unsqueeze(0)
        elif self.config.zero_init:
            h_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
            c_0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        else:
            h_0 = torch.rand(1, batch_size, self.hidden_size).cuda()
            c_0 = torch.rand(1, batch_size, self.hidden_size).cuda()
        
        return (h_0, c_0)


    def forward_backward(self, input_data, state_tuple):

        ## Forward pass
        hidden_states, state_tuple = self.lstm(input_data[:, :-1].transpose(0, 1), state_tuple)
        hidden_states = hidden_states.transpose(0, 1)

        ## Backward pass
        backw_input = torch.cat([hidden_states, input_data[:, 1:]], dim=-1)
        backw_input = backw_input.transpose(0, 1)
        neg_indexing = np.arange(len(backw_input))[::-1]
        backw_h, _ = self.backward_lstm(backw_input[list(neg_indexing)])
        backw_h = backw_h[list(neg_indexing)].transpose(0, 1)
        
        return hidden_states, backw_h, state_tuple


    def train_step(self, input_sequence, len_to_decode):

        ### Learning - a full sequence is given, posterior latent and reconstruction sequences are computed

        bs, length, _ = input_sequence.shape
        state_tuple = self.initialize_state_vectors(bs, input_sequence[:, 0, :self.output_size])

        x = input_sequence[..., :self.output_size].contiguous()
        v = input_sequence[..., self.output_size:-self.output_size].contiguous()
        a = input_sequence[..., -self.output_size:].contiguous()

        input_data = x.clone()
        if self.config.velocity_input:
            input_data = torch.cat([input_data, v], dim=-1)
        if self.config.acceleration_input:
            input_data = torch.cat([input_data, a], dim=-1)

        hidden_states, backw_h, _ = self.forward_backward(input_data, state_tuple)

        ## Recursive posterior sampling and reconstruction
        posterior_latent = self.init_z_layer(input_data[:, 0])
        reconstructed_out = []
        kl = 0

        for i in range(length - 1):

            h_n = hidden_states[:, i]

            # Posterior expectation of prior distribution
            statistics_p = self.prior(torch.cat([posterior_latent, h_n], dim=-1))
            mu_p, sigma_p = statistics_p[:, :self.hidden_size].contiguous(), statistics_p[:, self.hidden_size:].contiguous()
            sigma_p = torch.abs(sigma_p)

            # Posterior sampling
            statistics = self.posterior(torch.cat([posterior_latent, backw_h[:, i]], dim=-1))
            mu, sigma = statistics[:, :self.hidden_size].contiguous(), statistics[:, self.hidden_size:].contiguous()
            sigma = torch.abs(sigma)

            # Sample with reparameterization trick
            posterior_latent = mu + torch.randn_like(sigma) * sigma

            # Reconstruction
            statistics_x = self.conditional(torch.cat([posterior_latent, h_n], dim=-1))
            reconstructed_out.append(statistics_x)

            # KL divergence of two Gaussians
            kl += torch.log(torch.clamp(sigma_p / torch.clamp(sigma, min=1e-8), min=1e-8)).sum(dim=-1) - self.hidden_size * 0.5 + \
                0.5 * ((torch.clamp((sigma / torch.clamp(sigma_p, min=1e-8)), max=10) ** 2).sum(dim=-1) + (torch.clamp((mu - mu_p) / torch.clamp(sigma_p, min=1e-8), min=-10, max=10) ** 2).sum(dim=-1))
            
        kl  = kl / (i + 1)
        reconstructed_out = torch.stack(reconstructed_out, dim=1)
        mu_v, sigma_v = reconstructed_out[..., :self.output_size].contiguous(), reconstructed_out[..., self.output_size:].contiguous()
        sigma_v = torch.abs(sigma_v)
        mu_x = x[:, :-1] + mu_v
        mu_a = mu_v - v[:, :-1]
        reconstructed_out = torch.cat([mu_x, mu_v, mu_a], dim=-1)

        return reconstructed_out, sigma_v, kl


    def eval_step(self, input_sequence, len_to_decode):

        ### Synthesis - one / few initial samples are provided and the following is expected

        bs, length, _ = input_sequence.shape
        state_tuple = self.initialize_state_vectors(bs, input_sequence[:, 0, :self.output_size])

        x = input_sequence[..., :self.output_size].contiguous()
        v = input_sequence[..., self.output_size:-self.output_size].contiguous()
        a = input_sequence[..., -self.output_size:].contiguous()

        input_data = x.clone()
        if self.config.velocity_input:
            input_data = torch.cat([input_data, v], dim=-1)
        if self.config.acceleration_input:
            input_data = torch.cat([input_data, a], dim=-1)

        latent = self.init_z_layer(input_data[:, 0])

        if length > 0:
            hidden_states, backw_h, state_tuple = self.forward_backward(input_data, state_tuple)

            ## Recursive posterior sampling and reconstruction
            reconstructed_out = []

            for i in range(length - 1):

                h_n = hidden_states[:, i]

                # Posterior sampling
                statistics = self.posterior(torch.cat([latent, backw_h[:, i]], dim=-1))
                mu, sigma = statistics[:, :self.hidden_size].contiguous(), statistics[:, self.hidden_size:].contiguous()
                sigma = torch.abs(sigma)

                # Sample with reparameterization trick
                latent = mu + torch.randn_like(sigma) * sigma

                # Reconstruction
                statistics_x = self.conditional(torch.cat([latent, h_n], dim=-1))
                reconstructed_out.append(statistics_x)

            reconstructed_out = torch.stack(reconstructed_out, dim=1)
            mu_v, sigma_v = reconstructed_out[..., :self.output_size].contiguous(), reconstructed_out[..., self.output_size:].contiguous()
            sigma_v = torch.abs(sigma_v)
            mu_x = x[:, :-1] + mu_v
            mu_a = mu_v - v[:, :-1]
            reconstructed_out = torch.cat([mu_x, mu_v, mu_a], dim=-1)

        else:
            reconstructed_out = torch.empty(0, 0, 0).cuda()

        ## Synthesis
        generated_out = []
        last_pos = x[:, -1]
        last_vel = v[:, -1]
        last_acc = a[:, -1]

        for i in range(len_to_decode):
            
            # Forward one step lstm
            input_data = last_pos.clone()
            if self.config.velocity_input:
                input_data = torch.cat([input_data, last_vel], dim=-1)
            if self.config.acceleration_input:
                input_data = torch.cat([input_data, last_acc], dim=-1)
            output, (h_n, c_n) = self.lstm(input_data.unsqueeze(0), state_tuple)
            state_tuple = (h_n, c_n)

            # z sampling
            statistics_z = self.prior(torch.cat([latent, h_n.squeeze()], dim=-1))
            mu_z, sigma_z = statistics_z[:, :self.hidden_size].contiguous(), statistics_z[:, self.hidden_size:].contiguous()
            sigma_z = torch.abs(sigma_z)
            latent = mu_z + torch.randn_like(sigma_z) * sigma_z

            # x sampling TODO: --> non en fait pas de sample, on output le mean seulement et c'est ça qui devient x(t)
            statistics_x = self.conditional(torch.cat([latent, h_n.squeeze()], dim=-1))
            mu_x, sigma_x = statistics_x[:, :self.output_size].contiguous(), statistics_x[:, self.output_size:].contiguous()
            sigma_x = torch.abs(sigma_x)
            gen_output = mu_x + torch.randn_like(sigma_x) * sigma_x

            # Gen output is a displacement (instantaneous velocity)
            last_acc = gen_output - last_vel
            last_vel = gen_output
            last_pos += last_vel

            step_output = torch.cat([last_pos, last_vel, last_acc], dim=-1)
            generated_out.append(step_output)

        generated_out = torch.stack(generated_out, dim=1)

        return reconstructed_out, generated_out


class IRBTempMixer(nn.Module):
    """
    Inverted Residual Block for temporal mixing of sequences
    """

    def __init__(self, dim, expansion_fact):
        super(IRBTempMixer, self).__init__()
        self.pt_wise_in = nn.Conv1d(dim, expansion_fact * dim, 1)
        self.norm = nn.LayerNorm(expansion_fact * dim)
        self.depth_wise = nn.Conv1d(expansion_fact * dim, expansion_fact * dim, 3, padding=1, groups=expansion_fact * dim)
        self.act = nn.LeakyReLU()
        self.pt_wise_out = nn.Conv1d(expansion_fact * dim, dim, 1)
        
    def forward(self, x):
        # Expected shape: B x L x D, resp batch size, sequence length, dimension
        out = self.pt_wise_in(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm(out).transpose(1, 2)
        out = self.act(self.depth_wise(out))
        out = self.pt_wise_out(out).transpose(1, 2)
        return out

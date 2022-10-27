import h5py
import torch
import os
import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from torch.utils.data import Dataset
from .prepare_data import *


class Confer_Dataset(Dataset):

    ref_size = 180
    
    def __init__(self, base_dir, test=False, large=False, low_pass_filtering=0, dim=2, keypoints=False, jacobian=True, n_bands_melspec=5,
     fine_tuning_data=False, indep_temp_transform=False, max_length=100, test_files=[], range_high=0.8, range_low=0.6, mean_s=0.6, std_s=0.1, optimal_setting=False,
     rescaling_uniform=True, valid_index=0, validation=False):
        super(Confer_Dataset, self).__init__()

        if test:
            suffix = '_test'
            self.transform = False
        else:
            suffix = ''
            self.transform = True
        if large:
            suffix += '_p80_h0.5'
        
        self.dim = dim
        if dim == 3:
            prefix = '3D'
        else:
            prefix = ''

        self.range_high = range_high
        self.range_low = range_low
        self.optimal_setting = optimal_setting
        self.mean_s = mean_s
        self.std_s = std_s
        self.rescaling_uniform = rescaling_uniform

        self.low_pass_filtering = low_pass_filtering

        # Facial landmarks / unsupervised keypoints
        self.keypoints = keypoints
        self.jacobian = jacobian

        # Newer dataset version ?
        self.fine_tuning_dataset = fine_tuning_data
        self.n_bands_melspec = n_bands_melspec
        self.indep_temp_transform = indep_temp_transform

        groups = [ # divergence iteration
            ['20120305_seq1'], # 0, 30k, 50k
            ['20120514_seq7', '20120514_seq8', '20111212_seq1'], # 1, 45k, 30k
            ['20120326_seq3', '20111031_seq2', '20120319_seq5'], # 2, 30k
            ['20111031_seq1', '20111212_seq15', '20120507_seq2'], #3, None
            ['20120206_seq3', '20120604_seq1', '20120521_seq1', '20120604_seq15'], # 4, 49k
            ['20120604_seq10', '20120514_seq4', '20120123_seq4', '20120604_seq12', '20120123_seq2'], # 5, 30k
            ['20111205_seq1', '20120213_seq2', '20120514_seq5', '20120130_seq1', '20120423_seq11', '20120604_seq2'], # 6, 30k
            ['20111003_seq11', '20120213_seq1', '20120514_seq6', '20120423_seq4', '20120206_seq1', '20120604_seq9', '20120430_seq3'], # 7
            ['20120423_seq1', '20120423_seq10', '20111212_seq14', '20120521_seq8', '20120130_seq3', '20120206_seq2', '20120514_seq3'], # 8, 42k
            ['20120514_seq2', '20120430_seq5', '20120319_seq2', '20120430_seq12', '20120305_seq5', '20120604_seq11', '20120430_seq2', '20111212_seq4'], # 9, 52k
            ['20120423_seq6', '20111212_seq3', '20120423_seq2', '20120423_seq3', '20120206_seq4', '20120507_seq3', '20120507_seq5', '20120430_seq9', 
            '20120123_seq6', '20120430_seq10', '20120319_seq1'], # 10, 52k
            ['20120604_seq8', '20120423_seq9', '20120123_seq5', '20120123_seq3', '20120521_seq3', '20120604_seq4', '20111212_seq5', '20111205_seq2',
            '20120507_seq1', '20120123_seq8', '20120430_seq11', '20120430_seq7', '20120521_seq6'] # 11, 53k
        ]

        groups = {
            '3': ['20111031_seq1', '20111212_seq15', '20120507_seq2'],
            '7': ['20111003_seq11', '20120213_seq1', '20120514_seq6', '20120423_seq4', '20120206_seq1', '20120604_seq9', '20120430_seq3'],
            '31': ['20111031_seq1'], # 32k
            '32': ['20111212_seq15'],
            '33': ['20120507_seq2'],
            '34': ['20111212_seq15', '20120507_seq2'], # 29k
            '71': ['20111003_seq11', '20120213_seq1'], # 49k
            '72': ['20120514_seq6', '20120423_seq4'], # 50k
            '73': ['20120206_seq1', '20120604_seq9', '20120430_seq3'] # 55k
        }

        if self.fine_tuning_dataset:

            if max_length > 100:
                data_filename = f'landmark{prefix}_dataset_p{max_length}_h0.6_melspec'
            else:
                data_filename = f'landmark{prefix}_dataset_p100_h0.2_melspec'
            with open(os.path.join(base_dir, data_filename), 'rb') as f:
                self.dataset = pickle.load(f)
            if 'TEST_INDICES' in self.dataset.keys():
                self.dataset.pop('TEST_INDICES')
            # Load mel spectrograms
            self.melspec = self.dataset['melspec']
            self.dataset.pop('melspec')
            if len(test_files) == 0:
                with open(os.path.join(base_dir, 'test_files'), 'rb') as f:
                    test_files = pickle.load(f)
            test_samples = [k for k in self.dataset.keys() if k.split('#')[0] in test_files]
            if test:
                self.samples = test_samples
            else:
                valid_samples = [k for k in self.dataset.keys() if (str(valid_index) in groups.keys()) and (k.split('#')[0] in groups[str(valid_index)])]
                if validation:
                    self.samples = valid_samples
                else:
                    self.samples = [k for k in self.dataset.keys() if k not in valid_samples]
                # self.samples = [k for k in self.dataset.keys() if k not in test_samples]
            self.length = len(self.samples)

        elif self.keypoints:
            with open(os.path.join(base_dir, 'kp_dataset'), 'rb') as f:
                self.dataset = pickle.load(f)
            samples = ['#'.join((file, start_idx)) for file, value in self.dataset.items() for start_idx in value.keys()]
            with open(os.path.join(base_dir, 'kp_test_samples'), 'rb') as f:
                test_samples = pickle.load(f)
            if test:
                self.samples = test_samples
            else:
                samples = [sample for sample in samples if sample not in test_samples]
                valid_samples = [sample for sample in samples if (str(valid_index) in groups.keys()) and (sample.split('#')[0] in groups[str(valid_index)])]
                if validation:
                    self.samples = valid_samples
                else:
                    self.samples = [sample for sample in samples if sample not in valid_samples]
            self.length = len(self.samples)

        else:
            # type 1 error
            i_1 = [133, 140, 141, 203, 214, 264, 266, 284, 289, 303, 304, 306, 318, 319, 328, 330, 412, 413, 415, 492, 493, 494, 495, 496, 506,
            512, 530, 538, 594, 596, 598, 624, 632, 652, 658, 699, 704, 716, 717, 718, 719, 751, 762, 763, 793, 794, 795, 796, 797, 798, 
            804, 860, 861, 896, 916, 917, 960, 962, 990, 1016, 1046, 1047, 1048]
            # # type 2 error
            i_2 = [220, 221, 222, 223, 224, 225, 278, 279, 283, 329, 338, 339, 500, 508, 509, 560, 561, 608, 610, 627, 635, 641, 666, 667, 671,
            752, 913, 914, 915, 989, 1022]

            with h5py.File(os.path.join(base_dir, f'landmark{prefix}_dataset{suffix}.hdf5'), 'r') as f:
                self.dataset = f['landmark_db'][...]

            keys = [i for i in np.arange(len(self.dataset)) if ((i not in i_1) and (i not in i_2))]
            self.dataset = self.dataset[keys]

            self.length, self.n_pers, self.l, self.channels, _ = self.dataset.shape
    
    
    def __len__(self):
        return self.length
    
    
    def __getitem__(self, idx):

        if self.keypoints:
            # Fetch sample
            spl_name = self.samples[idx]
            file, start_idx = spl_name.split('#')
            spl_dict = self.dataset[file][start_idx]
            # Load kp, jacobians and bbox into separate tensors
            sample = np.stack([spl_dict[str(p_idx)]['keypoints']['value'] for p_idx in [1, 2]])
            jacobians = np.stack([spl_dict[str(p_idx)]['keypoints']['jacobian'] for p_idx in [1, 2]])
            bboxes = np.stack([spl_dict[str(p_idx)]['final_bbox'] for p_idx in [1, 2]])
            self.n_pers, self.l, self.channels, _ = sample.shape
            offset = 0
            spectro = np.zeros((1, self.l, 128, 1))
        else:
            if self.fine_tuning_dataset:
                spl_name = self.samples[idx]
                sample = self.dataset[spl_name] / self.ref_size
                self.n_pers, self.l, self.channels, _ = sample.shape # Un peu bizarre, à changer

                # Spectrogram
                spectro = self.melspec[spl_name]
                n_pads = int(0.5 * (self.n_bands_melspec - 1))
                spectro = np.pad(spectro, ((n_pads, n_pads), (0, 0), (0, 0)), constant_values=np.log(1e-8))
                # TODO: split le spectro après les augmtentations, qui sont faites sur l'ensemble, pê essayer d'abord sans low pass & dynamic range compression
                spectro = np.stack([np.concatenate([spectro[j] for j in range(i - n_pads, i + n_pads + 1)], axis=1) for i in range(n_pads, n_pads + self.l)])
                spectro = spectro[np.newaxis, :]

                # Max normalization
                spectro = spectro - np.amax(spectro, axis=(1, 2, 3), keepdims=True) + np.random.uniform(0.05, 0.75)

            else:
                spl_name = ''
                sample = self.dataset[idx] / self.ref_size
                spectro = np.zeros((1, self.l, 128, 1))
            bboxes = [0.0, 0.0, 0.0, 0.0]
            offset = 1

        if self.transform:

            # First, mb switch person order
            bool_switch = np.random.randint(2)
            if bool_switch == 1:
                sample = np.flip(sample, 0)

            # # Transformations can be made in two manners: independently for each time frame or not.
            # if self.indep_temp_transform:
            #     # Data augmentation is different in this case: separate rescaling factors are used for different time indices, and random translations are also applied
                
            #     ## Rescalings
            #     spatial_range = np.max(
            #         np.stack([
            #             sample[..., 0].max(axis=-1) - sample[..., 0].min(axis=-1), 
            #             sample[..., 1].max(axis=-1) - sample[..., 1].min(axis=-1)
            #         ]), axis=0
            #     )
            #     max_upscales = 0.99 / spatial_range
                
            #     rescalings = np.random.uniform(low=0.5, high=max_upscales, size=max_upscales.shape).clip(min=0.6) # TODO: lower limit dependent on size
            #     rescalings = rescalings.reshape(2 * self.l, 1)

            #     # Compute parameters of transformation matrix
            #     origin = 0.5 * (1 - rescalings) * np.ones(2)
            #     M = affine_matrix(rescalings, origin, 2)
                
            #     # Rescale and center
            #     sample = sample.reshape(2 * self.l, self.channels, 2)
            #     sample = scale_and_translate(sample, M).reshape(2, self.l, self.channels, 2)

            #     ## Translations
            #     bbox_left = sample[..., 0].min(axis=-1)
            #     bbox_right = sample[..., 0].max(axis=-1)
            #     bbox_top = sample[..., 1].min(axis=-1)
            #     bbox_bot = sample[..., 1].max(axis=-1)

            #     x_translations = np.random.uniform(low=0.01 - bbox_left, high=0.99 - bbox_right, size=bbox_left.shape)
            #     y_translations = np.random.uniform(low=0.01 - bbox_top, high=0.99 - bbox_bot, size=bbox_top.shape)

            #     sample[..., 0] = sample[..., 0] + x_translations[..., np.newaxis]
            #     sample[..., 1] = sample[..., 1] + y_translations[..., np.newaxis]

            #     ## Symmetries
            #     flips = np.random.randint(low=0, high=2, size=2 * self.l)
            #     sample = sample.reshape(2 * self.l, self.channels, 2)
            #     sample[np.where(flips == 1)[0], :, 0] =  offset - sample[np.where(flips == 1)[0], :, 0]
            #     sample = sample.reshape(2, self.l, self.channels, 2)

            #     ## Audio data augmentation

            #     bool_transform = np.random.randint(2, size=2)
            #     # Random dynamic range compression
            #     if bool_transform[0] == 1:
            #         mean = spectro.mean(axis=(2, 3), keepdims=True)
            #         std = spectro.std(axis=(2, 3), keepdims=True)
            #         rdm_thresh = np.random.uniform(0.2, 1.2, (1, spectro.shape[1], 1, 1))
            #         rdm_gain = np.random.uniform(0.5, 1, (1, spectro.shape[1], 1, 1))
            #         mask = (spectro > mean + rdm_thresh * std).astype(float)
            #         spectro = spectro * rdm_gain * mask + spectro * (1 - mask)
                    
            #     # Random low pass filter
            #     if bool_transform[1] == 1:
            #         freq_accuracy = 21.5
            #         cutting_freq = np.random.randint(50, 800)
            #         b, a = signal.butter(1, 2 * np.pi * cutting_freq, 'low', analog=True)
            #         w, h = signal.freqs(b, a, worN=2 * np.pi * (np.arange(spectro.shape[-2]) * freq_accuracy))
            #         # Addition of gain in log space
            #         mat = 3 * np.log10(np.clip(abs(h), 1e-9, None))[np.newaxis, np.newaxis, :, np.newaxis]
            #         spectro = spectro + mat

            # else:
            processed_p = []
            geom_range_y = (sample[..., 1].max(axis=-1) - sample[..., 1].min(axis=-1))
            geom_range_x = (sample[..., 0].max(axis=-1) - sample[..., 0].min(axis=-1))
            max_range = np.vstack([geom_range_y.max(axis=-1), geom_range_x.max(axis=-1)]).max(axis=0)
            mean_range = geom_range_y.mean(axis=-1)
            if self.rescaling_uniform:
                if self.optimal_setting:
                    max_upscales = 1.0
                else:
                    max_upscales = (self.range_high / mean_range)
                min_downscales = (self.range_low / mean_range)
                rescalings = np.random.uniform(low=min_downscales, high=max_upscales, size=2)
            else:
                rescalings = np.random.normal(loc=self.mean_s, scale=self.std_s, size=2).clip(min=0.4, max=0.9) / mean_range
            flips = np.random.randint(low=0, high=2, size=2)
            if self.dim == 3:
                # Draw final y orientation randomly in a fixed interval, then compute the required rotation around y-axis for each person
                z_i = (sample[..., 16, 2] - sample[..., 0, 2]).mean(axis=1)
                r = np.sqrt(((sample[..., 16, 2] - sample[..., 0, 2]) ** 2 + (sample[..., 16, 0] - sample[..., 0, 0]) ** 2)).mean(axis=1)
                theta_i = np.arcsin(-z_i / r)
                theta_y = np.random.uniform(low=-(np.pi / 3), high=np.pi / 3, size=2) - theta_i
                theta_x = np.random.uniform(low=-(np.pi / 12), high=np.pi / 12, size=2)

            # print(f'Idx: {idx}, transform params: {rescalings}, {flips}')

            for p_idx in [0, 1]:
                p_array = sample[p_idx]

                # Compute parameters of transformation matrix
                origin = np.zeros(2) if self.keypoints else (0.5 * (1 - rescalings[p_idx]) * np.ones(2))
                M = affine_matrix(rescalings[p_idx], origin, self.dim)

                # Rescale and center
                arr = p_array.reshape(self.l * self.channels, self.dim)
                arr = scale_and_translate(arr, M).reshape(self.l, self.channels, self.dim)

                # Random translation
                o_x = np.random.uniform(low=-arr[..., 0].min(), high=0.99 - arr[..., 0].max())
                o_y = np.random.uniform(low=-arr[..., 1].min(), high=0.99 - arr[..., 1].max())
                M = affine_matrix(1, [o_x, o_y], self.dim)
                arr = arr.reshape(self.l * self.channels, self.dim)
                arr = scale_and_translate(arr, M).reshape(self.l, self.channels, self.dim)

                # Rotate along y and x
                if self.dim == 3:
                    arr = rotate_3D(arr, theta_y[p_idx], theta_x[p_idx])

                # Maybe flip
                if flips[p_idx] == 1:
                    arr[..., 0] = offset - arr[..., 0]

                processed_p.append(arr)
            
            sample = np.stack(processed_p)

        if self.keypoints:
            sample = torch.Tensor(sample)
            jacobians = torch.Tensor(jacobians)
            if self.jacobian:
                sample = torch.cat([sample, jacobians.flatten(start_dim=-2)], dim=-1)
        else:
            sample = torch.Tensor(sample) # [..., :2]
        if self.low_pass_filtering > 0:
            sample = ma(sample, n=self.low_pass_filtering)
        out = [sample, torch.Tensor(spectro)[:, max(0, self.low_pass_filtering - 1):, ...], torch.Tensor(bboxes), spl_name]
            
        return out

class SEWA_Dataset(Dataset):

    ref_size = 180
    
    def __init__(self, base_dir, test=False, low_pass_filtering=0, dim=2, keypoints=False, jacobian=True, n_bands_melspec=5):

        data_filename = 'landmark_dataset_SEWA'
        with open(os.path.join(base_dir, data_filename), 'rb') as f:
            self.dataset = pickle.load(f)
        self.samples = list(self.dataset.keys())
        self.length = len(self.samples)
        self.transform = not test
        self.dim = dim
        self.low_pass_filtering = low_pass_filtering

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        spl_name = self.samples[idx]
        sample = self.dataset[spl_name] / self.ref_size
        self.n_pers, self.l, self.channels, _ = sample.shape

        if self.transform:
            offset = 1
            # First, mb switch person order
            bool_switch = np.random.randint(2)
            if bool_switch == 1:
                sample = np.flip(sample, 0)

            processed_p = []
            geom_range_y = (sample[..., 1].max(axis=-1) - sample[..., 1].min(axis=-1))
            geom_range_x = (sample[..., 0].max(axis=-1) - sample[..., 0].min(axis=-1))
            max_range = np.vstack([geom_range_y.max(axis=-1), geom_range_x.max(axis=-1)]).max(axis=0)
            mean_range = geom_range_y.mean(axis=-1)
            min_downscales = (0.4 / mean_range)
            rescalings = np.random.uniform(low=min_downscales, high=1.0, size=2)

            flips = np.random.randint(low=0, high=2, size=2)

            for p_idx in [0, 1]:
                p_array = sample[p_idx]

                # Compute parameters of transformation matrix
                origin = 0.5 * (1 - rescalings[p_idx]) * np.ones(2)
                M = affine_matrix(rescalings[p_idx], origin, self.dim)

                # Rescale and center
                arr = p_array.reshape(self.l * self.channels, self.dim)
                arr = scale_and_translate(arr, M).reshape(self.l, self.channels, self.dim)

                # Random translation
                o_x = np.random.uniform(low=-arr[..., 0].min(), high=0.99 - arr[..., 0].max())
                o_y = np.random.uniform(low=-arr[..., 1].min(), high=0.99 - arr[..., 1].max())
                M = affine_matrix(1, [o_x, o_y], self.dim)
                arr = arr.reshape(self.l * self.channels, self.dim)
                arr = scale_and_translate(arr, M).reshape(self.l, self.channels, self.dim)

                # Maybe flip
                if flips[p_idx] == 1:
                    arr[..., 0] = offset - arr[..., 0]

                processed_p.append(arr)
            
            sample = np.stack(processed_p)

        sample = torch.Tensor(sample)[..., :2]
        if self.low_pass_filtering > 0:
            sample = ma(sample, n=self.low_pass_filtering)
        out = [sample, spl_name]
            
        return out


def ma(a, n):
    '''
    Moving average on axis 1
    '''
    if n == 0:
        return a
    b = torch.cumsum(a, dim=1)
    b[:, n:] = b[:, n:] - b[:, :-n]
    return b[:, n - 1:] / n

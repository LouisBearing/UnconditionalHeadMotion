import torch
import os
import numpy as np
import pickle
import json
from torch.utils.data import Dataset
from scipy import signal
from .prepare_data import *


class VoxLipsDataset(Dataset):

    ref_size = 224
    
    def __init__(self, base_dir, test=False, low_pass_filtering=2, n_bands_melspec=3, subset_size=1.0):
        super(VoxLipsDataset, self).__init__()
        
        self.base_dir = base_dir
        self.transform = not test
        self.low_pass_filtering = low_pass_filtering
        self.n_bands_melspec = n_bands_melspec

        test_p_id = ['id00019', 'id00026', 'id00067', 'id00075', 'id00078', 'id00508', 'id00553', 'id00570', 'id00582', 'id00587']
    
        if test:
            person_id = test_p_id
        else:
            person_id = [_ for _ in os.listdir(base_dir) if _ not in test_p_id]
        fail_list = []
        if 'fail_list' in person_id:
            with open(os.path.join(base_dir, 'fail_list'), 'rb') as f:
                fail_list = pickle.load(f)
            person_id.remove('fail_list')

        self.vid_id = ['#'.join([p_id, '_'.join(f.split('_')[:-1])]) for p_id in person_id for f in os.listdir(os.path.join(base_dir, p_id)) if f.endswith('json')]
        self.vid_id = [_ for _ in self.vid_id if _ not in fail_list]

        if subset_size < 1:
            self.vid_id = np.random.choice(self.vid_id, size=int(len(self.vid_id) * subset_size), replace=False)


    def __len__(self):
        return len(self.vid_id)
    
    
    def __getitem__(self, idx):

        ### Landmarks
        p_id, v_id = self.vid_id[idx].split('#')
        with open(os.path.join(self.base_dir, p_id, v_id + '_coord.json'), 'rb') as f:
            ldks = np.array(json.load(f), dtype=int)
        
        if self.transform:
            # Random rescale & crop
            s_x, s_y = ldks.min(axis=(0, 1))
            w, h = ldks.max(axis=(0, 1)) - ldks.min(axis=(0, 1))
            s = max(w, h)
            factor = np.random.uniform(0.5, 0.9)
            crop_size = int(s / factor)
            crop_x = np.random.randint(s_x + w - crop_size + 1, s_x - 1)
            crop_y = np.random.randint(s_y + h - crop_size + 1, s_y - 1)
            rescaled_ldks = scale_and_translate(ldks.reshape(-1, 2), affine_matrix(1, -np.array([crop_x, crop_y]), 2)).reshape(ldks.shape)
            ldks = rescaled_ldks / crop_size

            flip = np.random.randint(low=0, high=2)
            if flip == 1:
                ldks[..., 0] = 1 - ldks[..., 0]

        else:
            ldks = ldks / self.ref_size

        # ### Spectrogram 
        # # sr video: 25 fps (same as ConferDB)
        # # sr audio: 16000 hz (22050 for ConferDB)
        # # res melspec: 0.016 s/pix
        # with open(os.path.join(self.base_dir, p_id, v_id + 'melspec'), 'rb') as f:
        #     melspec = pickle.load(f)
        
        # # Max normalization + random gain
        # melspec = melspec - np.max(melspec) + np.random.uniform(0.05, 0.75)

        # if self.transform:
        #     bool_transform = np.random.randint(2, size=2)
        #     # Random dynamic range compression
        #     if bool_transform[0] == 1:
        #         rdm_thresh = np.random.uniform(0.2, 1.2)
        #         rdm_gain = np.random.uniform(0.5, 1)
        #         mask = (melspec > melspec.mean() + rdm_thresh * melspec.std()).astype(float)
        #         melspec = melspec * rdm_gain * mask + melspec * (1 - mask)

        #     # Random low pass filter
        #     if bool_transform[1] == 1:
        #         freq_accuracy = 21.5
        #         cutting_freq = np.random.randint(50, 800)
        #         b, a = signal.butter(1, 2 * np.pi * cutting_freq, 'low', analog=True)
        #         w, h = signal.freqs(b, a, worN=2 * np.pi * (np.arange(melspec.shape[-2]) * freq_accuracy))
        #         # Addition of gain in log space
        #         mat = 3 * np.log10(np.clip(abs(h), 1e-9, None))[:, np.newaxis]
        #         melspec = melspec + mat

        # # Indices to split the spectrogram and synce with landmarks
        # pts_per_img = melspec.shape[1] / ldks.shape[0]
        # img_to_mel_idx = [int(np.round(idx * pts_per_img)) for idx in range(len(ldks))] + [melspec.shape[1]]
        # diff = np.diff(img_to_mel_idx)
        # img_to_mel_idx = [idx if (length == np.ceil(pts_per_img)) else (idx - 1) for idx, length in zip(img_to_mel_idx[:-1], diff)]

        # # Split spectrogram
        # n_pads = int(0.5 * (self.n_bands_melspec - 1) * np.ceil(pts_per_img))
        # melspec = np.pad(melspec, ((0, 0), (n_pads, n_pads)), constant_values=np.log(1e-8), mode='constant')
        # melspec = np.stack([melspec[:, i:i + 3 * n_pads] for i in img_to_mel_idx])

        # Convert to tensor
        sample = torch.Tensor(ldks)
        if self.low_pass_filtering > 0:
            sample = ma(sample, n=self.low_pass_filtering)
        # melspec = torch.Tensor(melspec)[max(0, self.low_pass_filtering - 1):, ...]

        melspec = torch.empty(0)

        return (sample, melspec)


def ma(a, n):
    '''
    Moving average on axis 0
    '''
    if n == 0:
        return a
    b = torch.cumsum(a, dim=0)
    b[n:] = b[n:] - b[:-n]
    return b[n - 1:] / n
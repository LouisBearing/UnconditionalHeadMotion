import os
import numpy as np
import json
import torch
import pandas as pd
import time
import pickle
from scipy.spatial.transform import Rotation as R


def affine_matrix(scale_factor, translation_vect, dim):
    if type(scale_factor) != np.ndarray:
        if dim == 3:
            translation_vect = np.concatenate([translation_vect, np.array([0])])
        M = np.hstack((scale_factor * np.eye(dim), np.array(translation_vect).reshape(dim, 1)))
        # Shape dim, 3
    else:
        # Then numpy array
        M = np.concatenate((scale_factor[..., np.newaxis] * (np.eye(dim)[np.newaxis, :]), translation_vect[..., np.newaxis]), axis=-1)
        # Shape 2 * length, dim, 3
    return M


def scale_and_translate(array, M):
    if len(array.shape) == 2:
        array = np.hstack([array, np.ones((len(array), 1))])
        affined_array = np.matmul(array, M.transpose())
    else:
        # Then batch operation
        array = np.concatenate([array, np.ones((array.shape[0], array.shape[1], 1))], axis=-1)
        affined_array = np.matmul(array, M.transpose(0, 2, 1))
    return affined_array


def RyRx_matrix(theta_x, theta_y):
    rx = R.from_quat([np.sin(theta_x / 2), 0, 0, np.cos(theta_x / 2)]).as_matrix()
    ry = R.from_quat([0, np.sin(theta_y / 2), 0, np.cos(theta_y / 2)]).as_matrix()
    return np.matmul(ry, rx)


def T_matrix(translation_vect):
    return np.hstack((np.eye(3), np.array(translation_vect).reshape(3, 1)))


def rotate_3D(np_array, theta_y, theta_x):
    '''
    np_array: shape seq_length, 68, 3
    '''
    origin = np_array.mean(axis=(0, 1))
    centered_arr = scale_and_translate(np_array.reshape(-1, 3), T_matrix(-origin))
    rotated_centered_arr = np.matmul(centered_arr, RyRx_matrix(theta_x, theta_y).transpose())
    rotated_arr = scale_and_translate(rotated_centered_arr, T_matrix(origin)).reshape(np_array.shape)
    return rotated_arr


def split_landmarks_seq(landmarks, dim=2, sewa=False):
    """
    Looks for discontinuities in the data and split sequences accordingly
    """

    sub_idx = np.array([], dtype=int)
    all_pts = []

    for idx in ['pers_1', 'pers_2']:

        if sewa:
            pts = landmarks[idx]

        else:
            # Fill null values with zeros. Those shall be deleted later on.
            for i in range(len(landmarks)):
                if len(landmarks[str(i)][idx]) == 0:
                    landmarks[str(i)][idx] = np.zeros((68, dim))
            pts = np.stack([np.array(landmarks[str(i)][idx]) for i in range(len(landmarks))]) # .astype(int)

        pts = interpolate(pts)
        all_pts.append(pts)

        # If all landmarks are concerned then this is a discontinuity
        new_sub_idx = get_discontinuity_indices(pts, thresh_displacement=10, thresh_count=67)
        sub_idx = np.union1d(sub_idx, new_sub_idx)

    all_pts = np.stack(all_pts)
    splits = np.split(all_pts, sub_idx, axis=1)
    splits = [(start_idx, split) for start_idx, split in zip(np.array([0] + sub_idx.tolist()), splits) if split.shape[1] > 1]
    
    return splits


def get_discontinuity_indices(pts, thresh_displacement, thresh_count):
    """
    Takes as input a sequence of landmark coordinates and outputs a list of discontinuity indices
    """

    # Distance between successive landmarks
    diff = np.diff(pts, axis=0)
    diff = np.sqrt((diff ** 2).sum(axis=2))

    # Counts number of points where the displacement exceeds threshold
    gaps = np.where(np.abs(diff) > thresh_displacement)[0] + 1
    un, counts = np.unique(gaps, return_counts=True)

    # Search how many landmark points are concerned
    sub_idx = un[np.where(counts > thresh_count)[0]]

    return sub_idx


def interpolate(pts, max_interp=2):

    sub_idx = get_discontinuity_indices(pts, thresh_displacement=15, thresh_count=50)

    # Find sequences of discontinuities: 1 frame long seq = actual discontinuity, 2 or more: landmark detection issues, then
    # interpolate if the sequence is short enough
    discontinuity_seq_idx = [0] + list(np.where(np.diff(sub_idx) > 2)[0] + 1) + [len(sub_idx)]
    discontinuity_seq = [sub_idx[i_0:i_f] for (i_0, i_f) in zip(discontinuity_seq_idx[:-1], discontinuity_seq_idx[1:])]

    # Keep only images where landmark detection failed
    discontinuity_seq = [seq for seq in discontinuity_seq if len(seq) > 1]

    # Interpolate between these indices
    interp_between = [[seq[0] - 1, seq[-1]] for seq in discontinuity_seq]
    interp_between = [seq for seq in interp_between if seq[1] - seq[0] - 1 <= max_interp]

    # Interpolate
    for (i_0, i_f) in interp_between:
        n_interp = i_f - i_0 - 1
        delta = pts[i_f] - pts[i_0]
        increment = delta / (n_interp + 1)
        
        buff = pts[i_0].copy()
        for i in np.arange(i_0 + 1, i_f):
            buff = buff + increment
            pts[i] = buff#.astype(int)

    return pts


def extend_samples_dataset(splits, ref_size, rescale, flip, l=125, p=100, h=0.2, dim=2):
    '''
    Samples l-length sequences from each element in splits. p is the duration of the sequence to predict and h an allowed
    overlap between ground truth prediction samples (dataset is normally built such that there is no overlap in the parts
    that must be predicted).
    '''

    processed_samples = []
    sample_info = []
    rescalings = []
    flips = []

    if rescale:
        scales = [[1, 1], [1, np.round(2 / 3, 3)], [np.round(2 / 3, 3), 1]]
    else:
        scales = [[1, 1]]


    for split_start_idx, split in splits:

        split_len, channels = split.shape[1], split.shape[2]

        # If any zero, skip
        if ~split.all():
            continue

        # Further divide into training samples
        start_idx = np.arange(0, split_len, int((1 - h) * p))
        samples = [(s_idx, split[:, s_idx:s_idx + l, ...]) for s_idx in start_idx if s_idx + l < split_len]

        for sample_start_idx, sample in samples:

            # Calculate the horizontal and vertical ranges of facial movements during the sequence, then find the appropriate
            # new reference frame origin (one new reference frame for each person in the interaction, hence possible different scales)
            spatial_extents = sample.max(axis=(1, 2)) - sample.min(axis=(1, 2))
            sizes = spatial_extents[:, :2].max(axis=1)
            sizes = (sizes * 1.1).astype(int)
            o_primes = ((sample.min(axis=(1, 2)).astype(int) + sample.max(axis=(1, 2)).astype(int) - sizes.reshape(2, 1)) / 2).astype(int)
            o_primes = o_primes[:, :2]
            
            for rescaling in scales:
                # Apply additional scaling factor to make training scale-independent
            
                processed_p = []
                inv_processed_p = []

                for p_idx in [0, 1]:

                    p_array = sample[p_idx]
                    S = sizes[p_idx]
                    o_prime = o_primes[p_idx]

                    # Compute parameters of transformation matrix
                    scale_f = ref_size / S
                    origin = (0.5 * ref_size * (1 - rescaling[p_idx]) - rescaling[p_idx] * scale_f * o_prime).astype(int)
                    M = affine_matrix(rescaling[p_idx] * scale_f, origin, dim)

                    arr = p_array.reshape(l * channels, dim)
                    arr = scale_and_translate(arr, M).reshape(l, channels, dim)

                    processed_p.append(arr)
                    
                    # Flip vertically
                    inv = arr.copy()
                    inv[..., 0] = ref_size - inv[..., 0]
                    inv_processed_p.append(inv)

                processed_samples.append(np.stack(processed_p))
                
                if flip:
                    mult_fact = 2
                    processed_samples.append(np.stack(inv_processed_p))
                    flips.extend([1, -1])
                else:
                    mult_fact = 1
                    flips.extend([1])
                sample_info.extend([split_start_idx + sample_start_idx] * mult_fact)
                rescalings.extend([rescaling] * mult_fact)
                
            
    return processed_samples, sample_info, rescalings, flips


def prepare_landmark_dataset(base_dir, out_dir, rescale, flip, l=125, p=100, h=0.2, ref_size=180, dim=2, test_idx=None):
    
    t = time.time()

    if dim == 3:
        prefix = '3D'
    else:
        prefix = ''

    suffix = f'_p{p}_h{h}'
    landmark_dir = f'write_json_{prefix}landmarks'
    melspec_dir = os.path.join(base_dir, 'write_melspec')
    with open(os.path.join(melspec_dir, 'melspec'), 'rb') as f:
        melspecs = pickle.load(f)

    dataset = {}
    dataset['melspec'] = {}

    for file in os.listdir(os.path.join(base_dir, landmark_dir)):
        filename = file.split('.')[0]

        landmark_path = os.path.join(base_dir, landmark_dir, file)

        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)

        melspec = melspecs[filename]
        pts_per_img = melspec.shape[1] / len(landmarks)
        # Find pix indices in the spectrogram that correspond to video images - one image is uniquely related to ~3 melspec pixels
        img_to_mel_idx = [int(np.round(idx * pts_per_img)) for idx in range(len(landmarks))] + [melspec.shape[1]]
        diff = np.diff(img_to_mel_idx)
        img_to_mel_idx = [idx if (length == np.ceil(pts_per_img)) else (idx - 1) for idx, length in zip(img_to_mel_idx[:-1], diff)]
        # Pad first column
        melspec = np.pad(melspec, ((0, 0), (1, 0)), mode='reflect')
        # Split mel spectrogram per image and stack the result in a (n_frames x n_mel_coeff x 3) tensor
        melspec = np.stack([melspec[:, (idx + 1):(idx + 1) + np.ceil(pts_per_img).astype(int)] for idx in img_to_mel_idx])

        splits = split_landmarks_seq(landmarks, dim=dim)
        
        new_processed_samples, new_sample_info, _, _ = extend_samples_dataset(splits, ref_size, rescale=rescale, flip=flip, l=l, p=p, h=h, dim=dim)
        new_sample_info = ['#'.join((filename, str(s_idx))) for s_idx in new_sample_info]
        dataset.update({sample_name: new_processed_samples[i] for i, sample_name in enumerate(new_sample_info)})

        dataset['melspec'].update({
            sample_name: melspec[int(sample_name.split('#')[-1]):int(sample_name.split('#')[-1]) + l] for sample_name in new_sample_info
        })

        print(f'File {file} done in {time.time() - t}')

    # # Train test split
    # if test_idx is None:
    #     length = len(dataset)
    #     test_idx = np.random.choice(list(dataset.keys()), int(0.05 * length), replace=False)
    # dataset['TEST_INDICES'] = test_idx

    with open(os.path.join(out_dir, f'landmark{prefix}_dataset{suffix}_melspec'), 'wb') as f:
        pickle.dump(dataset, f)
    
    # with h5py.File(os.path.join(base_dir, f'cf_landmark{prefix}_dataset{suffix}.hdf5'), 'w') as f:
    #     f.create_dataset('landmark_db', data=train_dataset)

    # with h5py.File(os.path.join(base_dir, f'landmark{prefix}_dataset_test{suffix}.hdf5'), 'w') as f:
    #     f.create_dataset('landmark_db', data=test_dataset)
    
    # train_ds_info.to_csv(os.path.join(base_dir, f'{prefix}dataset_info.csv'), index=False)
    # test_ds_info.to_csv(os.path.join(base_dir, f'{prefix}dataset_info_test.csv'), index=False)

def prepare_SEWA_dataset(landmark_dir, out_dir, l=50, p=40, h=0, ref_size=180, dim=2):
    
    t = time.time()

    dataset = {}

    for filename in os.listdir(landmark_dir):
        landmark_path = os.path.join(landmark_dir, filename)

        with open(landmark_path, 'rb') as f:
            landmarks = pickle.load(f)

        splits = split_landmarks_seq(landmarks, dim=dim, sewa=True)
        
        new_processed_samples, new_sample_info, _, _ = extend_samples_dataset(splits, ref_size, rescale=False, flip=False, l=l, p=p, h=h, dim=dim)
        new_sample_info = ['#'.join((filename, str(s_idx))) for s_idx in new_sample_info]
        dataset.update({sample_name: new_processed_samples[i] for i, sample_name in enumerate(new_sample_info)})

        print(f'File {filename} done in {time.time() - t}')

    with open(os.path.join(out_dir, f'landmark_dataset_SEWA'), 'wb') as f:
        pickle.dump(dataset, f)
    
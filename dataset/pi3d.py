'''
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
MIT license.
'''
# pi3d.py

import torch
from torch.utils.data import Dataset
import numpy as np
# from IPython import embed

class Pi3d_Dataset(Dataset):

    def __init__(self, opt, path_to_data, is_train=True):

        self.path_to_data = path_to_data
        self.is_train = is_train
        if is_train:#train
            self.in_n = opt.input_n
            self.out_n = opt.kernel_size
            self.split = 0
        else: #test
            self.in_n = 50
            self.out_n = opt.output_n
            self.split = 1
        self.skip_rate = 1
        self.p3d = {}
        self.data_idx = []

        if opt.protocol == 'pro3': # unseen action split
            if is_train: #train on acro2
                acts = ["2/a-frame","2/around-the-back","2/coochie","2/frog-classic","2/noser","2/toss-out", "2/cartwheel",\
                        "1/a-frame","1/around-the-back","1/coochie","1/frog-classic","1/noser","1/toss-out", "1/cartwheel"]
                subfix = [[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[1,2,3,4,5],[2,3,4,5,6],\
                        [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,4,5,6],[1,2,3,4,6],[1,2,3,4,5],[3,4,5,6,7]]

            else: #test on acro1
                acts = ["2/crunch-toast", "2/frog-kick", "2/ninja-kick", \
                        "1/back-flip", "1/big-ben", "1/chandelle", "1/check-the-change", "1/frog-turn", "1/twisted-toss"]
                subfix = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],\
                        [1,3,4,5,6],[1,2,3,4,5],[3,4,5,6,7],[1,2,4,5,8],[1,2,3,4,5],[1,2,3,4,5]]

                if opt.test_split is not None: #test per action for unseen action split
                    acts, subfix = [acts[opt.test_split]], [subfix[opt.test_split]]

        else: # common action split and single action split
            if is_train: #train on acro2
                acts = ["2/a-frame","2/around-the-back","2/coochie","2/frog-classic","2/noser","2/toss-out", "2/cartwheel"]
                subfix = [[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[1,2,3,4,5],[2,3,4,5,6]]

                if opt.protocol in ["0","1","2","3","4","5","6"]: # train per action for single action split
                    acts = [acts[int(opt.protocol)]]
                    subfix = [subfix[int(opt.protocol)]]

            else: #test on acro1
                acts = ["1/a-frame","1/around-the-back","1/coochie","1/frog-classic","1/noser","1/toss-out", "1/cartwheel"]
                subfix = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,4,5,6],[1,2,3,4,6],[1,2,3,4,5],[3,4,5,6,7]]

                if opt.test_split is not None: #test per action for common action split
                    acts, subfix = [acts[opt.test_split]], [subfix[opt.test_split]]
                if opt.protocol in ["0","1","2","3","4","5","6"]: #test per action for single action split
                    acts, subfix = [acts[int(opt.protocol)]], [subfix[int(opt.protocol)]]

        key = 0
        for action_idx in np.arange(len(acts)):
            subj_action = acts[action_idx]
            subj, action = subj_action.split('/')
            for subact_i in np.arange(len(subfix[action_idx])):
                subact = subfix[action_idx][subact_i]
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                filename = '{0}/acro{1}/{2}{3}/mocap_cleaned.tsv'.format(self.path_to_data, subj, action, subact)
                the_sequence = readCSVasFloat(filename,with_key=True)
                num_frames = the_sequence.shape[0]
                the_sequence = normExPI_2p_by_frame(the_sequence)
                the_sequence = torch.from_numpy(the_sequence).float().cuda()

                if self.is_train: #train
                    seq_len = self.in_n + self.out_n
                    valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)
                else: #test
                    seq_len = self.in_n + 30
                    valid_frames = find_indices_64(num_frames, seq_len)

                p3d = the_sequence
                self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                tmp_data_idx_1 = [key] * len(valid_frames)
                tmp_data_idx_2 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                key += 1

        self.dimension_use = np.arange(18*2*3)
        self.in_features = len(self.dimension_use)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        data = self.p3d[key][fs][:,self.dimension_use]
        return data


###########################################
## func for reading data

def readCSVasFloat(filename, with_key=True):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    if with_key: # skip first line
        lines = lines[1:]
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray

###########################################
## func utils for norm/unnorm

def normExPI_xoz(img, P0,P1,P2):
    # P0: orig
    # P0-P1: axis x
    # P0-P1-P2: olane xoz

    X0 = P0
    X1 = (P1-P0) / np.linalg.norm((P1-P0)) + P0 #x
    X2 = (P2-P0) / np.linalg.norm((P2-P0)) + P0
    X3 = np.cross(X2-P0, X1-P0) + P0 #y
    ### x2 determine z -> x2 determine plane xoz
    X2 = np.cross(X1-P0, X3-P0) + P0 #z

    X = np.concatenate((np.array([X0,X1,X2,X3]).transpose(), np.array([[1, 1, 1,1]])), axis = 0)
    Q = np.array([[0,0,0],[1,0,0],[0,0,1], [0,1,0]]).transpose()
    M  = Q.dot(np.linalg.pinv(X))

    img_norm = img.copy()
    for i in range(len(img)):
        tmp = img[i]
        tmp = np.concatenate((tmp,np.array([1])),axis=0)
        img_norm[i] =  M.dot(tmp)
    return img_norm

def normExPI_2p_by_frame(seq):
    nb, dim = seq.shape # nb_frames, dim=108
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1,3)) #36
        P0 = (img[10] + img[11])/2
        P1 = img[11]
        P2 = img[3]
        img_norm = normExPI_xoz(img, P0,P1,P2)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm

def unnorm_abs2Indep(seq):
    # in:  torch.size(bz, nb_frames, 36, 3)
    # out: torch.size(bz, nb_frames, 36, 3)
    seq = seq.detach().cpu().numpy()
    bz, frame, nb, dim = seq.shape
    seq_norm = seq
    for j in range(bz):
        for i in range(frame):
            img = seq[j][i]

            P0_m = (img[10] + img[11])/2
            P1_m = img[11]
            P2_m = img[3]
            if nb == 36:
                img_norm_m = normExPI_xoz(img[:int(nb/2)], P0_m,P1_m,P2_m)
                P0_f = (img[18+10] + img[18+11])/2
                P1_f = img[18+11]
                P2_f = img[18+3]
                img_norm_f = normExPI_xoz(img[int(nb/2):], P0_f,P1_f,P2_f)
                img_norm = np.concatenate((img_norm_m, img_norm_f))
            elif nb == 18:
                img_norm = normExPI_xoz(img, P0_m,P1_m,P2_m)
            seq_norm[j][i] = img_norm.reshape((nb,dim))
    seq = torch.from_numpy(seq_norm).cuda()
    return seq


###########################################
## func utils for finding test samples

def find_indices_64(num_frames, seq_len):
    # not random choose. as the sequence is short and we want the test set to represent the seq better
    seed = 1234567890
    np.random.seed(seed)

    T = num_frames - seq_len + 1
    n = int(T / 64)
    list0 = np.arange(0,T)
    list1 = np.arange(0,T,(n+1))
    t =  64 - len(list1)
    if t == 0:
        listf = list1
    else:
        list2 = np.setdiff1d(list0, list1)
        list2 = list2[:t]
        listf = np.concatenate((list1, list2))
    return listf
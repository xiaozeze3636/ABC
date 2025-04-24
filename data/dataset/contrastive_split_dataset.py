import os
import torch
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import Counter
from tqdm import tqdm
from logging import getLogger
from libcity.data.dataset import BaseDataset, padding_mask, TrajectoryProcessingDataset
from libcity.data.dataset.bertlm_dataset import noise_mask
import random
import math

# 数据增强函数定义
def random_deletion(sequence, p=0.1):
    if len(sequence) == 1:
        return sequence
    new_sequence = []
    for item in sequence:
        r = random.uniform(0, 1)
        if r > p:
            new_sequence.append(item)
    if len(new_sequence) == 0:
        new_sequence.append(random.choice(sequence))
    return new_sequence

def random_swap(sequence, n_swaps=1):
    new_sequence = sequence.copy()
    for _ in range(n_swaps):
        idx1 = random.randint(0, len(new_sequence)-1)
        idx2 = random.randint(0, len(new_sequence)-1)
        new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]
    return new_sequence

def random_insertion(sequence, n_insertions=1):
    new_sequence = sequence.copy()
    for _ in range(n_insertions):
        insert_item = random.choice(new_sequence)
        insert_idx = random.randint(0, len(new_sequence))
        new_sequence.insert(insert_idx, insert_item)
    return new_sequence

def random_masking(sequence, vocab, p=0.15):
    new_sequence = []
    mask_index = vocab.mask_index
    for item in sequence:
        r = random.uniform(0, 1)
        if r < p:
            new_sequence.append(mask_index)
        else:
            new_sequence.append(item)
    return new_sequence

def time_shifting(sequence, max_shift=5, pad_index=0):
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return sequence
    elif shift > 0:
        return [pad_index]*shift + sequence[:-shift]
    else:
        return sequence[-shift:] + [pad_index]*(-shift)

def add_noise(sequence, sigma=0.05):
    """在轨迹中添加高斯噪声"""
    noisy_sequence = []
    for item in sequence:
        noisy_item = item + np.random.normal(0, sigma, size=item.shape)
        noisy_sequence.append(noisy_item)
    return noisy_sequence

def scaling(sequence, scale_factor=1.1):
    """缩放轨迹"""
    scaled_sequence = [item * scale_factor for item in sequence]
    return scaled_sequence

def rotation(sequence, angle=15):
    """旋转轨迹"""
    radians = math.radians(angle)
    rotation_matrix = np.array([[math.cos(radians), -math.sin(radians)],
                                [math.sin(radians), math.cos(radians)]])
    rotated_sequence = [rotation_matrix.dot(item[:2]).tolist() + item[2:] for item in sequence]
    return rotated_sequence

# 数据集类定义
class ContrastiveSplitDataset(BaseDataset):
    # breakpoint()
    def __init__(self, config):
        super().__init__(config)
        self.argu1 = config.get('out_data_argument1', None)
        self.argu2 = config.get('out_data_argument2', None)
        self.data_argument1 = self.config.get("data_argument1", [])
        self.data_argument2 = self.config.get("data_argument2", [])
        self.masking_ratio = self.config.get('masking_ratio', 0.2)
        self.masking_mode = self.config.get('masking_mode', 'together')
        self.distribution = self.config.get('distribution', 'random')
        self.avg_mask_len = self.config.get('avg_mask_len', 3)
        self.collate_fn = collate_unsuperv_contrastive_split

    def _gen_dataset(self):
        train_dataset = TrajectoryProcessingDatasetSplit(
            data_name=self.dataset, data_type='train', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=self.max_train_size,
            argu1=self.argu1, argu2=self.argu2,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio,
            masking_mode=self.masking_mode,
            distribution=self.distribution,
            avg_mask_len=self.avg_mask_len)
        eval_dataset = TrajectoryProcessingDatasetSplit(
            data_name=self.dataset, data_type='eval', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            argu1=self.argu1, argu2=self.argu2,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio,
            masking_mode=self.masking_mode,
            distribution=self.distribution,
            avg_mask_len=self.avg_mask_len)
        test_dataset = TrajectoryProcessingDatasetSplit(
            data_name=self.dataset, data_type='test', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            argu1=self.argu1, argu2=self.argu2,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio,
            masking_mode=self.masking_mode,
            distribution=self.distribution,
            avg_mask_len=self.avg_mask_len)
        return train_dataset, eval_dataset, test_dataset

class TrajectoryProcessingDatasetSplit(TrajectoryProcessingDataset):
    # breakpoint()

    def __init__(self, data_name, data_type, vocab, seq_len=512, add_cls=True,
                 merge=True, min_freq=1, max_train_size=None, argu1=None, argu2=None,
                 data_argument1=None, data_argument2=None,
                 masking_ratio=0.2, masking_mode='together',
                 distribution='random', avg_mask_len=3):

        self.vocab = vocab
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self._logger = getLogger()
        self._logger.info('Init TrajectoryProcessingDatasetSplit!')

        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.distribution = distribution
        self.avg_mask_len = avg_mask_len
        self.exclude_feats = None
        # breakpoint()
        # 手动指定数据增强方法列表
        self.data_argument1 = ['random_deletion', 'random_swap', 'add_noise', 'random_masking']
        self.data_argument2 = ['random_insertion', 'time_shifting', 'scaling', 'rotation']

        # 设置增强方法的参数
        self.random_deletion_p = 0.1
        self.random_swap_n = 1
        self.random_insertion_n = 1
        self.random_masking_p = 0.15
        self.time_shifting_max_shift = 5
        self.add_noise_sigma = 0.05
        self.scaling_factor = 1.1
        self.rotation_angle = 15  # degrees

        # 如果需要调整数据增强方法列表，可以通过参数传入
        self.data_argument1 = data_argument1 if data_argument1 is not None else self.data_argument1
        self.data_argument2 = data_argument2 if data_argument2 is not None else self.data_argument2

        if 'mask' in self.data_argument1:
            self._logger.info('Use mask as data argument in view1!')
        if 'mask' in self.data_argument2:
            self._logger.info('Use mask as data argument in view2!')

        # 数据路径设置
        if argu1 is not None:
            self.data_path1 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu1)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu1)
        else:
            self.data_path1 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)
        if argu2 is not None:
            self.data_path2 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu2)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu2)
        else:
            self.data_path2 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)

        self.temporal_mat_path1 = self.cache_path1[:-4] + '_temporal_mat.pkl'
        self.temporal_mat_path2 = self.cache_path2[:-4] + '_temporal_mat.pkl'
        self._load_data()

    def _load_data(self):
        print(f"Cache Path1: {self.cache_path1}")
        print(f"Temporal Mat Path1: {self.temporal_mat_path1}")
        print(f"Cache Path2: {self.cache_path2}")
        print(f"Temporal Mat Path2: {self.temporal_mat_path2}")

        if os.path.exists(self.cache_path1) and os.path.exists(self.temporal_mat_path1) \
                and os.path.exists(self.cache_path2) and os.path.exists(self.temporal_mat_path2):
            self.traj_list1 = pickle.load(open(self.cache_path1, 'rb'))
            self.temporal_mat_list1 = pickle.load(open(self.temporal_mat_path1, 'rb'))
            self.traj_list2 = pickle.load(open(self.cache_path2, 'rb'))
            self.temporal_mat_list2 = pickle.load(open(self.temporal_mat_path2, 'rb'))
            self._logger.info('Load dataset from {}, {}'.format(self.cache_path1, self.cache_path2))
            print(111111111111111)
        else:
            print(211111111111111)
            origin_data_df1 = pd.read_csv(self.data_path1, sep=';')
            origin_data_df2 = pd.read_csv(self.data_path2, sep=';')
            assert origin_data_df1.shape == origin_data_df2.shape
            self.traj_list1, self.temporal_mat_list1 = self.data_processing(
                origin_data_df1, self.data_path1, cache_path=self.cache_path1, tmat_path=self.temporal_mat_path1)
            self.traj_list2, self.temporal_mat_list2 = self.data_processing(
                origin_data_df2, self.data_path2, cache_path=self.cache_path2, tmat_path=self.temporal_mat_path2)
        if self.max_train_size is not None:
            self.traj_list1 = self.traj_list1[:self.max_train_size]
            self.temporal_mat_list1 = self.temporal_mat_list1[:self.max_train_size]
            self.traj_list2 = self.traj_list2[:self.max_train_size]
            self.temporal_mat_list2 = self.temporal_mat_list2[:self.max_train_size]

    def __len__(self):
        assert len(self.traj_list1) == len(self.traj_list2)
        return len(self.traj_list1)

    def __getitem__(self, ind):
        traj_ind1 = self.traj_list1[ind]  # (seq_length, feat_dim)
        print(traj_ind1.shape)
        traj_ind2 = self.traj_list2[ind]  # (seq_length, feat_dim)
        temporal_mat1 = self.temporal_mat_list1[ind]  # (seq_length, seq_length)
        temporal_mat2 = self.temporal_mat_list2[ind]  # (seq_length, seq_length)
        mask1 = None
        mask2 = None
        breakpoint()

        # 对traj_ind1应用数据增强
        if 'random_deletion' in self.data_argument1:
            traj_ind1 = random_deletion(traj_ind1, p=self.random_deletion_p)
        if 'random_swap' in self.data_argument1:
            traj_ind1 = random_swap(traj_ind1, n_swaps=self.random_swap_n)
        if 'add_noise' in self.data_argument1:
            traj_ind1 = add_noise(traj_ind1, sigma=self.add_noise_sigma)
        if 'scaling' in self.data_argument1:
            traj_ind1 = scaling(traj_ind1, scale_factor=self.scaling_factor)
        if 'rotation' in self.data_argument1:
            traj_ind1 = rotation(traj_ind1, angle=self.rotation_angle)
        if 'random_masking' in self.data_argument1:
            traj_ind1 = random_masking(traj_ind1, self.vocab, p=self.random_masking_p)
        if 'time_shifting' in self.data_argument1:
            traj_ind1 = time_shifting(traj_ind1, max_shift=self.time_shifting_max_shift, pad_index=self.vocab.pad_index)

        # 对traj_ind2应用数据增强
        if 'random_deletion' in self.data_argument2:
            traj_ind2 = random_deletion(traj_ind2, p=self.random_deletion_p)
        if 'random_swap' in self.data_argument2:
            traj_ind2 = random_swap(traj_ind2, n_swaps=self.random_swap_n)
        if 'add_noise' in self.data_argument2:
            traj_ind2 = add_noise(traj_ind2, sigma=self.add_noise_sigma)
        if 'scaling' in self.data_argument2:
            traj_ind2 = scaling(traj_ind2, scale_factor=self.scaling_factor)
        if 'rotation' in self.data_argument2:
            traj_ind2 = rotation(traj_ind2, angle=self.rotation_angle)
        if 'random_insertion' in self.data_argument2:
            traj_ind2 = random_insertion(traj_ind2, n_insertions=self.random_insertion_n)
        if 'time_shifting' in self.data_argument2:
            traj_ind2 = time_shifting(traj_ind2, max_shift=self.time_shifting_max_shift, pad_index=self.vocab.pad_index)
        if 'random_masking' in self.data_argument2:
            traj_ind2 = random_masking(traj_ind2, self.vocab, p=self.random_masking_p)

        # 如果需要使用遮盖（mask）数据增强
        if 'mask' in self.data_argument1:
            mask1 = noise_mask(traj_ind1, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        if 'mask' in self.data_argument2:
            mask2 = noise_mask(traj_ind2, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        print(traj_ind1)
        return torch.LongTensor(traj_ind1), torch.LongTensor(traj_ind2), \
               torch.LongTensor(temporal_mat1), torch.LongTensor(temporal_mat2), \
               torch.LongTensor(mask1) if mask1 is not None else None, \
               torch.LongTensor(mask2) if mask2 is not None else None

# 辅助函数定义
def _inner_slove_data(features, temporal_mat, batch_size, max_len, vocab=None, mask=None):
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
        if mask[i] is not None:
            target_masks[i, :end, :] = mask[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    if mask[0] is not None:
        X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
        X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    return X, batch_temporal_mat, padding_masks

# 批处理函数定义
def collate_unsuperv_contrastive_split(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features1, features2, temporal_mat1, temporal_mat2, mask1, mask2 = zip(*data)  # list of (seq_length, feat_dim)
    lengths1 = [len(f) for f in features1]
    lengths2 = [len(f) for f in features2]
    max_len1 = max(lengths1)
    max_len2 = max(lengths2)
    max_len = max(max_len1, max_len2) if max_len is None else max_len

    X1, batch_temporal_mat1, padding_masks1 = _inner_slove_data(
        features1, temporal_mat1, batch_size, max_len, vocab, mask1)
    X2, batch_temporal_mat2, padding_masks2 = _inner_slove_data(
        features2, temporal_mat2, batch_size, max_len, vocab, mask2)
    return X1.long(), X2.long(), padding_masks1, padding_masks2, \
           batch_temporal_mat1.long(), batch_temporal_mat2.long()

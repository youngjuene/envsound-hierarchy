import os
import json
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from config import FILELIST_DIR, MELSPEC_DIR

# Set the appropriate paths of the datasets here.

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label_list in enumerate(labels):
        for label in label_list:
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

    return label2inds


def load_data(file):
    with open(FILELIST_DIR+file, 'rb') as fo:
        data = pickle.load(fo)
    return data


class FSD_MIX_CLIPS(Dataset):
    def __init__(self, phase='train', multilabel=False):
        assert(phase=='train' or phase=='val' or phase=='test')
        self.phase = phase
        self.multilabel = multilabel

        print('Loading FSD_MIX_CLIPS {0} dataset.'.format(phase))
        file_train_categories_train_phase = 'base_train_filelist.pkl'
        file_train_categories_val_phase = 'base_val_filelist.pkl'
        file_train_categories_test_phase = 'base_test_filelist.pkl'
        file_val_categories_val_phase = 'val_filelist.pkl'
        file_test_categories_test_phase = 'test_filelist.pkl'
        mel_dir = MELSPEC_DIR

        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

            mel_p = [mel_dir + ('/').join(p.replace('.wav', '.npy').split('/')[-3:]) for p in self.data]
            self.mel_path = [p.replace('base/', 'base_') for p in mel_p]

            self.x, self.y = self.pair(input_data_list = self.mel_path, 
                                       label_list = self.labels)

        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # evaluates the recognition accuracy of the base categories.
                data_base = load_data(file_train_categories_test_phase)
                # evaluates the few-shot recogniton accuracy on the novel categories.
                data_novel = load_data(file_test_categories_test_phase)
            else: # phase=='val'
                # evaluates the recognition accuracy of the base categories.
                data_base = load_data(file_train_categories_val_phase)
                # evaluates the few-shot recogniton accuracy on the novel categories.
                data_novel = load_data(file_val_categories_val_phase)

            self.data = np.concatenate([data_base['data'], data_novel['data']], axis=0)
            base_mel_p = [mel_dir + ('/').join(p.replace('.wav', '.npy').split('/')[-3:]) for p in data_base['data']]
            self.base_mel_path = [p.replace('base/', 'base_') for p in base_mel_p]
            self.novel_mel_path = [mel_dir + ('/').join(p.replace('.wav', '.npy').split('/')[-2:]) for p in data_novel['data']]
            self.mel_path = self.base_mel_path + self.novel_mel_path
            self.labels = data_base['labels'] + data_novel['labels']
            self.x, self.y = self.pair(
                input_data_list = self.mel_path, 
                label_list = self.labels
                )

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = buildLabelIndex(data_base['labels']).keys()
            self.labelIds_novel = buildLabelIndex(data_novel['labels']).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)

        else:
            raise ValueError('Phase {0} not valid.'.format(self.phase))


    def __getitem__(self, index):
        path, label = self.x[index], self.y[index]
        mel = np.load(path)
        return mel, label

    def __len__(self):
        return len(self.x)

    def pair(self, input_data_list, label_list):
        x = []
        y = []
        for i in range(len(input_data_list)):
            # single: 351781, multi: 96342, total: 448123
            if self.multilabel == False:  # single label 
                if len(label_list[i]) == 1:
                    x.append(input_data_list[i])
                    y.append(label_list[i][0])
            else: # more than two label, includes single label events
                if len(label_list[i]) != 1: 
                    x.append(input_data_list[i])
                    y.append(label_list[i])
                else:
                    x.append(input_data_list[i])
                    y.append(label_list[i][0])
        return x, y

class PrototypicalBatchSampler:
    """
    Yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'n_class' and 'num_samples',
    Every iteration, the batch indexes (n_support + n_query) samples for 'n_classes' random classes.
    
    __len__????????? epoch??? ?????????????????? ????????????. (self.iterations??? ??????.)
    """
    def __init__(self, labels, n_class, num_samples, iterations):
        """
        Args:
        - labels: an iterable containing all the labels for the current dataset. ?????? ???????????????????????? ??? sample??? index?????? ???????????? ?????? ??????.
        - n_class: ??? iteration?????? ????????? ?????????????????? ??? (K-way)
        - num_samples: ??? iteration????????? ??? ?????????????????? ????????? ??? (support+query)
        - iterations: ??? epoch ??? iteration??? ???(= episode??? ???)
        """
        super().__init__()
        self.labels = labels
        self.n_class = n_class
        self.sample_per_class = num_samples
        self.iterations = iterations
        
        self.classes, self.counts = np.unique(self.labels, return_counts=True) # numpy ?????? ??? ????????? ????????? ????????? ??????, ??? ????????? ????????? ????????? ????????? ????????? ??????
        self.classes = torch.LongTensor(self.classes)
        
        # ???????????? x max(????????? ??? ?????????) ????????? ?????????.
        # nan?????? ?????? ???, ?????? ?????? ?????? ??? ????????? c??? ???????????? ???????????? ?????????.
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.num_per_class = torch.zeros_like(self.classes)  # ??? ???/?????? ?????? ????????? ????????? ?????? ?????? 0?????? ????????? ?????? ??????
        for idx, label in enumerate(self.labels): # ?????? ?????? ????????????.
            label_idx = np.argwhere(self.classes == label).item()  # self.classes ????????????, ???????????? ???????????? ????????? ????????? ??? ?????? ???????????? ??????(????????? ?????? ??????)
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx  # label_idx??? ???????????????, nan????????? ???????????? ???????????? ????????? ?????? idx??? ????????????
            self.num_per_class[label_idx] += 1  # ???????????? ?????? ????????? ????????????
            
    def __iter__(self):
        """
        yield a batch of indexes
        """
        cpi = self.n_class
        spc = self.sample_per_class

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi] # ????????? 39??? ??? ??? 12?????? ?????????
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.num_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch
            
    def __len__(self):
        """
        1 epoch??? episode ??? ??????
        ????????? ???????????? ????????? batch??????????????? (class * (support + query), 1, 128, 126)??? ???.
        """
        return self.iterations


def get_loader(phase,
               n_class,
               n_shot,
               n_query,
               multilabel_bool=False):
    dataset = FSD_MIX_CLIPS(phase, multilabel=multilabel_bool)
    sample_size = n_shot + n_query
    dataloader = DataLoader(
        dataset,
        batch_sampler=PrototypicalBatchSampler(
            labels=dataset.y,
            n_class=n_class,
            num_samples=sample_size,
            iterations = len(dataset.y)//(n_class*sample_size)),
            num_workers=16)
    return dataloader, dataset
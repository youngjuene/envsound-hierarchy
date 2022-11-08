from audioop import mul
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import reduce
from operator import __add__
from tree import MusicTree
from typing import List, Dict
from collections import OrderedDict



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding='same'):
        super().__init__()
        assert isinstance(kernel_size, tuple)

        if padding == 'same' and stride == 1:
            padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        else:
            raise ValueError('not implemented anything other than same padding and stride 1')

        self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.b_norm = nn.BatchNorm2d(1) # (batch, C, H, W)에서 C를 입력
        self.conv1 = ConvBlock(in_channels=1, out_channels=128, kernel_size=(3, 3))
        self.conv2 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv3 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv4 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.linear = nn.Linear(1024, 128)
        
    def forward(self, x): # input (192, 128, 126)
        # (batch, 1, frequency, time) i.g. (192, 1, 128, 126)
        x = x.unsqueeze(1)
        x = self.b_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        d_time = x.shape[-1]
        x = F.max_pool2d(x, kernel_size=(1, d_time))

        # (batch, feature)로 reshape
        d_batch = x.shape[0]
        x = x.view(d_batch, -1)
        x = self.linear(x)
        
        return x


class HierarchicalProtoNet(pl.LightningModule):
    def __init__(self,
                 n_class, n_shot, n_query, 
                 height,
                 loss_alpha,
                 taxonomy,
                 lr):
        super().__init__()
        self.n_class = n_class
        self.n_shot = n_shot
        self.n_query = n_query

        assert height >= 0
        self.height = height

        if taxonomy == 'AS':
            taxonomy = yaml.safe_load(open("./taxonomy/AS.yaml", 'r'))
        elif taxonomy == 'VR':
            taxonomy = yaml.safe_load(open("./taxonomy/VR.yaml", 'r'))
        elif taxonomy == 'SG':
            taxonomy = yaml.safe_load(open("./taxonomy/SG.yaml", 'r'))

        self.loss_alpha = loss_alpha
        self.lr = lr
        self.loss_weight_fn = 'exp'
        self.loss_beta = 0.5
        self.tree = MusicTree.from_taxonomy(taxonomy)
        self.tree.even_depth()
                
        self.save_hyperparameters()

    def split_batch(self, mels, targets, n_class=5, n_shot=5, n_query=10):
        _, idx = torch.sort(targets)
        idx = idx.view(n_class, -1)

        s_indices = torch.tensor([list(range(n_shot))] * n_class)
        q_indices = torch.tensor([list(range(n_shot,n_query+n_shot))] * n_class)

        sup = torch.gather(idx.detach().cpu(), 1, s_indices)
        qry = torch.gather(idx.detach().cpu(), 1, q_indices)
        
        sup = sup.view(-1).tolist()
        qry = qry.view(-1).tolist()

        support_mels    = torch.stack([mels[s]    for s in sup])
        support_targets = torch.stack([targets[s] for s in sup])

        query_mels    = torch.stack([mels[q]    for q in qry])
        query_targets = torch.stack([targets[q] for q in qry])

        return support_mels, support_targets, query_mels, query_targets

    """ LEAF LOSS """
    def calculate_prototypes(self, features, targets):
        classes, _ = torch.unique(targets).sort()
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, classes

    def compute_leaf_loss(self, prototypes, classes, feats, targets):
        dists = torch.cdist(feats.unsqueeze(0), prototypes.unsqueeze(0), p=2)
        assert dists.shape[0] == 1
        dists = dists[0]
        preds = -dists
        labels = (classes[None, :] == targets[:, None]).int().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        loss = F.cross_entropy(preds, labels)

        h0_loss = dict()
        h0_loss['preds'] = preds.argmax(dim=1)
        h0_loss['labels'] = labels
        h0_loss['loss'] = loss
        h0_loss['acc'] = acc
        h0_loss['height'] = 0
        return h0_loss
    
    """ ANCESTOR LOSS """
    def compute_ancestor_prototypes(self, prototypes, classes, height: int):
        """ take care """
        # prototypes, classes = prototypes.detach().cpu(), classes.detach().cpu()
        
        # ordered dictionary of prototypes
        # we will use this to group prototypes by ancestor later
        protos = OrderedDict([(n, t) for n, t in zip(classes, prototypes)])

        # get the unique list of ancestors,
        # as well as dict w format class: ancestor
        ancestor_classlist, class2ancestor = self.get_ancestor_classlist(classes, height)

        # store ancestor prototypes here
        ancestor_protos = OrderedDict()
        for tgt in classes:
            # get this class's ancestor
            classname = classdict[tgt.item()]
            ancestor = class2ancestor[classname]

            # if we have already registered this ancestor,
            # it means we already grabbed the relevant tensors for it,
            # so skip it
            if ancestor in ancestor_protos:
                continue

            # otherwise, grab all the tensors that share this ancestor
            # torch.Size([1, 128])
            tensor_stack = torch.stack([p for n, p in protos.items() if class2ancestor[classdict[n.item()]] == ancestor])

            # take the prototype of the prototypes!
            # torch.Size([128])
            ancestor_proto = tensor_stack.mean(dim=0, keepdim=False)

            # register this prototype with the rest of ancestor protos
            ancestor_protos[ancestor] = ancestor_proto

        # convert the OrderedDict into a tensor for our loss function
        # I should probably rearrange ancestor_protos here to match ancestor_classlist
        ancestor_protos = [ancestor_protos[a] for a in ancestor_classlist]
        ancestor_protos = torch.stack(ancestor_protos) 

        return ancestor_protos # torch.Size([12, 128])

    def compute_ancestor_losses(self, prototypes, classes, query_feats, query_targets):
        total_height = self.tree.depth()
        h_loss = []
        for height in range(1, total_height):
            ancestor_classlist, class2ancestor = self.get_ancestor_classlist(query_targets, height)
            ancestor_protos = self.compute_ancestor_prototypes(prototypes, classes, height=height)
            ancestor_targets = self.get_ancestor_targets(ancestor_classlist, query_targets, class2ancestor).type_as(ancestor_protos).long()

            # May 4th, 'type_as' also changes tensor's device!  https://discuss.pytorch.org/t/type-as-also-changes-tensors-device/41991
            # ancestor_dists = torch.pow(ancestor_protos[None,:].type_as(query_feats) - query_feats[:, None], 2).sum(dim=2)
            ancestor_dists = torch.cdist(query_feats.unsqueeze(0), ancestor_protos.unsqueeze(0).type_as(query_feats), p=2)
            ancestor_dists = ancestor_dists.squeeze(0)
            preds = -ancestor_dists # torch.Size([144, 9])

            # ancestor_dists torch.Size([1, 144, 9]), 9 when h=1
            # ancestor_dists = F.log_softmax(-ancestor_dists, dim=1) 
            # ancestor_dists = ancestor_dists.squeeze(0)
            # preds = torch.argmax(ancestor_dists, dim=-1, keepdim=False)

            labels = ancestor_targets.type_as(preds).long()
            acc = (preds.argmax(dim=1) == labels).float().mean() # 분류오류
            loss = F.cross_entropy(preds, labels)
            
            height_dict = dict()
            height_dict['preds'] = preds
            height_dict['labels'] = labels
            height_dict['loss'] = loss
            height_dict['acc'] = acc
            height_dict['height'] = height
            h_loss.append(height_dict)
        return h_loss

    def compute_both_losses(self, batch, mode):
        # GPUtil.showUtilization()

        # Determine training loss for a given support set and query set
        mels, targets = batch
        # features.shape = (225, 128); 225 = (n5*(k15+q30))
        features = self.model(mels)  # Encode all support/query data
        support_feats, support_targets, query_feats, query_targets = self.split_batch(features, targets, n_class=self.n_class, n_shot=self.n_shot, n_query=self.n_query)
        del features, targets # free up space held by these tensors
        
        prototypes, classes = self.calculate_prototypes(support_feats, support_targets)
        del support_feats, support_targets
        
        """ LEAF LOSS """
        leaf_loss = self.compute_leaf_loss(prototypes,
                                           classes,
                                           query_feats,
                                           query_targets)
        
        """ ANCESTOR LOSS """
        ancestor_loss = self.compute_ancestor_losses(prototypes, 
                                                     classes,
                                                     query_feats,
                                                     query_targets)

        return leaf_loss, ancestor_loss

    def compute_losses(self, batch, mode):
        leaf_loss, ancestor_loss = self.compute_both_losses(batch, mode)
        meta_loss = [leaf_loss] + ancestor_loss  # [dict] + list
        loss_vec = torch.stack([t['loss'] for t in meta_loss])

        output = dict()
        if self.height > 0:
            if self.loss_weight_fn == 'exp': # 1.0  0.4724  0.2231
                self.loss_weights = torch.exp(-self.loss_alpha * torch.arange(self.height+1)) 
                if self.height == 1:
                    self.loss_weights = torch.cat([self.loss_weights, torch.zeros(1)])
                output['loss'] = torch.sum(self.loss_weights.type_as(loss_vec) * loss_vec)
            # 'exp-leafheavy'
            elif self.loss_weight_fn == 'exp-coarse': # 1.0  2.7183  7.3891
                self.loss_weights = torch.exp(self.loss_alpha * torch.arange(self.height+1))
                if self.height == 1:
                    self.loss_weights = torch.cat([self.loss_weights, torch.zeros(1)])
                output['loss'] = torch.sum(self.loss_weights.type_as(loss_vec) * loss_vec)

            elif self.loss_weight_fn == "interp-avg":
                self.loss_weights = torch.ones(self.height) / (self.height-1) * (1-self.loss_alpha)
                self.loss_weights[0] = 1 * self.loss_alpha
                output['loss'] = self.loss_alpha * loss_vec[0] + (1 - self.loss_alpha) * torch.mean(loss_vec[1:])

            elif self.loss_weight_fn == "interp-avg-decay":
                # alpha should be 0.5-1.0 (1 for baseline, 0 for all hierarchical)
                # beta should be 0.75, 1, 1.25, (0 for interp-avg)

                #  beta is an exponential decay factor for tree losses
                # alpha is a linear interpolation factor for mixing tree loss with leaf loss
                self.loss_weights = torch.exp(-self.loss_beta * torch.arange(self.height-1, -1, -1)) * (1-self.loss_alpha)
                self.loss_weights[0] = torch.tensor(1) * self.loss_alpha
                self.loss_weights = self.loss_weights.type_as(loss_vec)

                output['loss'] = self.loss_alpha * loss_vec[0] + torch.mean(loss_vec[1:] * self.loss_weights[1:])

            elif self.loss_weight_fn == "cross-entropy":
                self.loss_weights = torch.tensor([1, 0, 0, 0])
                output['loss'] = self.hierarchy_multi_hot(meta_loss)

            else:
                raise ValueError

        else:  # height = 0; non-hierarchical
            self.loss_weights = torch.tensor([1, 0, 0]) # coefficient for h=0, h=1, h=2
            output['loss'] = meta_loss[0]['loss']
            output['acc'] = meta_loss[0]['acc']  # accuracy는 아직..

        # insert metatasks in ascending order
        output['tasks'] = meta_loss

        output['loss-weights'] = self.loss_weights

        return output

    def get_ancestor_classlist(self, targets, height=0):
        classlist = [classdict[labelid.item()] for labelid in targets]
        # get a 1-to-1 mapping from the parents to ancestors
        # get the list of parents for each of the members in the classlist
        ancestors = [self.tree.get_ancestor(c, height) for c in classlist]

        # note that ancestor_classlist is a UNIQUE set of strings,
        # and should thus be used to generate numeric labels for the ancestors
        ancestor_classlist = list(set(ancestors))

        # map from fine grained classes to our ancestors
        c2a = OrderedDict([(c, a) for c, a in zip(classlist, ancestors)])

        return ancestor_classlist, c2a

    def get_ancestor_targets(self, ancestor_classlist, query_targets, c2a):
        return torch.tensor([ancestor_classlist.index(c2a[classdict[l.item()]]) for l in query_targets]) 
    
    def training_step(self, batch, batch_idx):
        output = self.compute_losses(batch, mode="train")
        self.log("train_loss", output['loss'], on_step=True, on_epoch=False)
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.compute_losses(batch, mode="val")
        self.log("val_loss", output['loss'], on_step=False, on_epoch=True)

        return {"val_loss": output['loss']}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                               lr = self.lr,
                               weight_decay=1e-5)

        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=1000,
                verbose=True,
            ),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

def batch_detach_cpu(x):
    """syntax honey"""
    return batch_cpu(batch_detach(x))

def batch_detach(nested_collection):
    """ move a dict of tensors to detach. 
    no op if tensors already in detach
    """
    if isinstance(nested_collection, dict):
        for k, v in nested_collection.items():
            if isinstance(v, torch.Tensor):
                nested_collection[k] = v.detach()
            if isinstance(v, dict):
                nested_collection[k] = batch_detach(v)
            elif isinstance(v, list):
                nested_collection[k] = batch_detach(v)
    if isinstance(nested_collection, list):
        for i, v in enumerate(nested_collection):
            if isinstance(v, torch.Tensor):
                nested_collection[i] = v.detach()
            elif isinstance(v, dict):
                nested_collection[i] = batch_detach(v)
            elif isinstance(v, list):
                nested_collection[i] = batch_detach(v)
    return nested_collection

def batch_cpu(nested_collection):
    """ move a dict of tensors to cpu. 
    no op if tensors already in cpu
    """
    if isinstance(nested_collection, dict):
        for k, v in nested_collection.items():
            if isinstance(v, torch.Tensor):
                nested_collection[k] = v.cpu()
            if isinstance(v, dict):
                nested_collection[k] = batch_cpu(v)
            elif isinstance(v, list):
                nested_collection[k] = batch_cpu(v)
    if isinstance(nested_collection, list):
        for i, v in enumerate(nested_collection):
            if isinstance(v, torch.Tensor):
                nested_collection[i] = v.cpu()
            elif isinstance(v, dict):
                nested_collection[i] = batch_cpu(v)
            elif isinstance(v, list):
                nested_collection[i] = batch_cpu(v)
    return nested_collection
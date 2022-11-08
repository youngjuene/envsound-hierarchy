import os
import glob
import time
import pickle
import warnings
from zlib import Z_DEFAULT_COMPRESSION
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean, stdev
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn 

from data import get_loader
from protonet import HierarchicalProtoNet

def get_file_name_dict(dataset_fnames):
    npy_fname_dict = {}
    for fname in dataset_fnames:
        npy = np.load(fname)
        npy_fname_dict[npy.tobytes()] = fname
    return npy_fname_dict

@torch.no_grad()
def test_proto_net(model, 
                   n_class,
                   k_shot, 
                   query_length,
                   save_f1,
                   data_feats=None):
                   
    model.eval()
    if data_feats is None:
        dataloader, test_set = get_loader(phase='test',
                                          n_class=n_class, 
                                          n_shot=k_shot,
                                          n_query=query_length,
                                          multilabel_bool=False)

    else:
        mel_features, mel_targets = data_feats
    npy_fname_dict = get_file_name_dict(test_set.x)

    f1_list = []
    cm_labels, cm_preds = [], []

    print(f"Evaluation on {k_shot} support sample case.")
    for loader_idx, (x, y) in enumerate(tqdm(dataloader,
                                            "Evaluating prototype classification ...", 
                                            leave=False)):

        x = x.float().to(device)
        y = y.int()

        _, idx = y.sort()
        support_y_idx, query_y_idx = torch.split(idx.view(n_class, -1), 
                                                 [k_shot, query_length], 
                                                 dim=1)
        support_x = torch.stack([x[i] for i in torch.flatten(support_y_idx)])
        support_y = torch.stack([y[i] for i in torch.flatten(support_y_idx)])
        query_x = torch.stack([x[i] for i in torch.flatten(query_y_idx)])
        query_y = torch.stack([y[i] for i in torch.flatten(query_y_idx)])
        x_s = model.module.model(support_x).detach().cpu()
        x_q = model.module.model(query_x).detach().cpu()
        prototypes, proto_classes = model.module.calculate_prototypes(x_s, support_y)
        h0_loss = model.module.compute_leaf_loss(prototypes, proto_classes, x_q, query_y)
        preds, labels, acc = h0_loss['preds'], h0_loss['labels'], h0_loss['acc']
        sound_dict = dict(zip(range(len(torch.unique(query_y))), torch.unique(query_y).tolist()))

        for l in labels.tolist():
            cm_labels.append(sound_dict[l])
        for p in preds.tolist():
            cm_preds.append(sound_dict[p])
        
        
        embedding_of_this_episode = {}
        support_fname = [npy_fname_dict[x.cpu().numpy()[i].tobytes()] for i in torch.flatten(support_y_idx).tolist()]
        embedding_of_this_episode['support_set'] = {'mel_npy': support_fname, 
                                                    'support_emb': x_s, 
                                                    'target': support_y} 
        query_fname = [npy_fname_dict[x.cpu().numpy()[i].tobytes()] for i in torch.flatten(query_y_idx).tolist()]
        embedding_of_this_episode['query_set'] = {'mel_npy': query_fname,
                                                  'query_emb': x_q, 
                                                  'target': query_y}
        save_path = SAVE_RESULT_DIR+f'emb/{n_class}-way/{k_shot}-shot/{query_length}-query/'
        os.makedirs(save_path, exist_ok=True)
        torch.save(embedding_of_this_episode, save_path+f'episode_{loader_idx}.pt')
        
        f1 = f1_score(labels.tolist(), preds.tolist(), average='micro', zero_division=0)
        f1_list.append(f1)

    # Saving Results.
    f1_dict = dict()
    f1_dict[k_shot] = (f1_list, mean(f1_list), stdev(f1_list))
    f1_df = pd.DataFrame({
        "F1 Score": 
            f1_dict[k_shot],
        "Number of Support Examples":
            [k_shot]*len(f1_dict[k_shot]),
        "Query length[s]":
            [query_length]*len(f1_dict[k_shot])
    })
    save2pkl((cm_labels, cm_preds), SAVE_RESULT_DIR+save_f1, 'conf', n_class, k_shot, query_length)
    savef1pkl(f1_df, SAVE_RESULT_DIR, save_f1=save_f1)
    # return 


def save2pkl(input_data, pkl_save_path, score_type, n, k, q):
    os.makedirs(pkl_save_path+f'/{score_type}', exist_ok=True)
    with open(pkl_save_path+f'/{score_type}/{n}way{k}shot{q}query.pkl', 'wb') as f:
        pickle.dump(input_data, f)

def savef1pkl(input_data, pkl_save_path, save_f1):
    os.makedirs(pkl_save_path+f'/', exist_ok=True)
    with open(pkl_save_path+f'/{save_f1}.pkl', 'wb') as f:
        pickle.dump(input_data, f)

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    date = time.strftime('%Y-%m-%d_%H:%M', time.localtime(time.time()))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    parser = argparse.ArgumentParser(description='Parser for non-/hierarchical prototypical networks.')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--way', type=int, required=True, default=5)
    parser.add_argument('--shot', type=int, required=True, default=15)
    parser.add_argument('--query', type=int, required=True, default=30)
    parser.add_argument('--taxonomy', type=str, required=True, default='AS', help='Provide taxonomy among AS, VR, SG.')
    parser.add_argument('--alpha', type=float, default=-0.5)
    parser.add_argument('--lr', type=float, default=1e-03)

    args = parser.parse_args()
    
    n_class = args.way
    n_shot = args.shot
    n_query = args.query
    height = args.height
    taxonomy = args.taxonomy
    loss_alpha = args.alpha
    lr = args.lr


    for taxonomy in enumerate(['AS', 'VR', 'SG']):
        ckptlist = glob.glob(f'./exp/{taxonomy}/*')
        best_ckptlist = [glob.glob(i+'/Best*') for i in ckptlist]
        best_ckpts = [item for sublist in best_ckptlist for item in sublist]
        'H{height}_n{n_class}k{n_shot}q{n_query}_a{loss_alpha}_lr{lr}_{date}'
        'H0 n5k15q30 a-0.25 lr0.001'
        for best_ckpt in best_ckpts:
            opt = best_ckpt.split('/')[-2].split('_')[:-1]
            SAVE_RESULT_DIR = f"./result/{opt[1]}/{opt[2]}/{taxonomy}/{height}/"
            f1_path = f'{taxonomy}_{opt[0]}_{opt[2]}'
            
            model = HierarchicalProtoNet.load_from_checkpoint(best_ckpt)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[0,1]).cuda()

            test_proto_net(model,
                           n_class=n_class,
                           k_shot=n_shot,
                           query_length=n_query,
                           save_f1=f1_path)

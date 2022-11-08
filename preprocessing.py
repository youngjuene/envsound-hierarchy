import os
import json
import pickle
import librosa
import numpy as np
from tqdm import tqdm

wav_dir = './FSD-MIX-CLIPS/'
save_mel_dir = './data/mel/'
data_split = ['base/train/', 'base/val/', 'base/test/', 'val/', 'test/']
filelist = ['base_train_filelist.pkl', 'base_val_filelist.pkl', 'base_test_filelist.pkl',
            'val_filelist.pkl', 'test_filelist.pkl']

for f in filelist:
    pklfile = pickle.load(open(wav_dir+'filelist/'+f, 'rb'))
    wav_path = pklfile['data']
    label = pklfile['labels']

    tag = f[:-13]
    for idx, wpath in tqdm(enumerate(wav_path)):
        save_dir = save_mel_dir+tag
        fname = wpath.split('/')[-1].replace('.wav', '.npy')
        mel_save_path = save_dir+'/'+fname
        if not os.path.exists(mel_save_path):
            os.makedirs(save_dir, exist_ok=True)
            y, sr = librosa.load(wpath, sr=16000)
            assert sr == 16000

            spec = librosa.feature.melspectrogram(
                y=y, sr=sr,
                win_length=16*32,
                hop_length=16*8,
                power=1)

            np.save(mel_save_path, spec)
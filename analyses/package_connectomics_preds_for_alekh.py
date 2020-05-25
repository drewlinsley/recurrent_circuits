import os
import numpy as np
from utils import py_utils
from ops import tf_fun
from glob import glob
from tqdm import tqdm


p = '/media/data_cifs/cluster_projects/refactor_gammanet/neurips_data'
e = 'for_alekh'
dirs = [
    'berson_001',
    'berson_010',
    'berson_100',
    'snemi_001',
    'snemi_010',
    'snemi_100',
    'seung_berson_001',
    'seung_berson_010',
    'seung_berson_100',
    'seung_snemi_001',
    'seung_snemi_010',
    'seung_snemi_100',
]


for d in tqdm(dirs, total=len(dirs)):
    z = np.load(glob(os.path.join(p, d, '*.npz'))[0])['test_dict']
    out_path = os.path.join(p, e, d)
    label_path = os.path.join(out_path, 'groundTruth')
    pred_path = os.path.join(out_path, 'predictions')
    py_utils.make_dir(out_path)
    py_utils.make_dir(label_path)
    py_utils.make_dir(pred_path)
    for idx, i in enumerate(z):
        label = i['labels']
        pred = tf_fun.sigmoid_fun(i['logits'])
        np.save(os.path.join(label_path, '%s.npy' % idx), label)
        np.save(os.path.join(pred_path, '%s.npy' % idx), pred)

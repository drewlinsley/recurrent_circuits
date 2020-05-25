import os
import numpy as np
from tqdm import tqdm
from glob import glob


def sigmoid_fun(x):
    """Apply sigmoid to maps before mAP."""
    return 1 / (1 + np.exp(x))


out_dirs = [
    '/media/data_cifs/cluster_projects/neurips_data/berson_100_aug_proc',
    '/media/data_cifs/cluster_projects/neurips_data/snemi_100_aug_proc',
    '/media/data_cifs/cluster_projects/neurips_data/seung_berson_100_aug_proc',
    '/media/data_cifs/cluster_projects/neurips_data/seung_snemi_100_aug_proc',
]
files = [
    '/media/data_cifs/cluster_projects/neurips_data/berson_100_aug',
    '/media/data_cifs/cluster_projects/neurips_data/snemi_100_aug',
    '/media/data_cifs/cluster_projects/neurips_data/seung_berson_100_aug',
    '/media/data_cifs/cluster_projects/neurips_data/seung_snemi_100_aug',
]

for f, o in tqdm(zip(files, out_dirs), total=len(files)):
    if not os.path.exists(o):
        os.makedirs(o)
    f = glob(os.path.join(f, '*npz'))[0]
    out_path = os.path.join(o, f.split(os.path.sep)[-1].split('_')[0])
    val_dict = np.load(f, encoding='latin1')['test_dict']
    preds, labs = [], []
    for d in val_dict:
        preds += [sigmoid_fun(d['logits'])]
        labs += [d['labels']]
    preds = np.array(preds)
    labs = np.array(labs)
    np.save('%s_preds' % out_path, preds)
    np.save('%s_labs' % out_path, labs)

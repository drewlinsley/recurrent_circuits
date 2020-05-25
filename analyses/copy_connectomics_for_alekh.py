import os
import numpy as np
from glob import glob
from tqdm import tqdm


def sigmoid_fun(x):
    """Apply sigmoid to maps before mAP."""
    return 1 / (1 + np.exp(x))


out_dirs = [
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_snemi_005_seung_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_snemi_005_gammanet_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_berson_005_seung_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_berson_005_gammanet_v3',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_snemi_010_seung_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_snemi_010_gammanet_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_berson_010_seung_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_berson_010_gammanet_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_snemi_100_seung_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_snemi_100_gammanet_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_berson_100_seung_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/redo_berson_100_gammanet_v2',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/snemi_100A_seung',
    # '/media/data_cifs/cluster_projects/neurips_data/maps/berson_100A_seung',
    '/media/data_cifs/cluster_projects/neurips_data/maps/snemi_100Av2_gammanet',
    '/media/data_cifs/cluster_projects/neurips_data/maps/berson_100Av2_gammanet',
]
files = [
    # '/media/data_cifs/cluster_projects/refactor_gammanet/snemi_100A/seung_unet_per_pixel_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/berson_100A/seung_unet_per_pixel_*',
    '/media/data_cifs/cluster_projects/refactor_gammanet/snemi_100Av2/refactored_v7_snemi_*',
    '/media/data_cifs/cluster_projects/refactor_gammanet/berson_100Av2/refactored_v7_berson_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_snemi_005/seung_unet_per_pixel_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_snemi_005/refactored_v7_snemi_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_berson_005/seung_unet_per_pixel_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_berson_005/refactored_v7_berson_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_berson_005/refactored_v7_berson_combos_test_2019_11_08_17_02_40_860944.npz',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_snemi_010/seung_unet_per_pixel_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_snemi_010/refactored_v7_snemi_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_berson_010/seung_unet_per_pixel_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_berson_010/refactored_v7_berson_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_snemi_100/seung_unet_per_pixel_snemi_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_snemi_100/refactored_v7_snemi_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_berson_100/seung_unet_per_pixel_*',
    # '/media/data_cifs/cluster_projects/refactor_gammanet/redo_berson_100/refactored_v7_berson_*',
]
for f, o in tqdm(zip(files, out_dirs), total=len(files)):
    f = glob(f)[0]
    out_path = os.path.join(o, f.split(os.path.sep)[-1].split('_')[0])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_dict = np.load(f, allow_pickle=True)['test_dict']
    preds, labs = [], []
    for d in val_dict:
        preds += [sigmoid_fun(d['logits'])]
        labs += [d['labels']]
    preds = np.array(preds)
    labs = np.array(labs)
    np.save('%s_preds' % out_path, preds)
    np.save('%s_labs' % out_path, labs)

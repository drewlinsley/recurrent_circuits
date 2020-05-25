import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'BSDS500_100_jk',
    ]
    exp['val_dataset'] = [
        'BSDS500_100_jk',
    ]
    exp['model'] = [
    ]
    exp['validation_period'] = [100]
    exp['validation_steps'] = [100]
    exp['shuffle_val'] = [False]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]
    # exp['exclusion_scope'] = 'fgru'

    # Model hyperparameters
    exp['lr'] = [1e-3]
    exp['loss_function'] = ['impute_bi_bce']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']
    exp['score_function'] = ['pixel_error']  # ['pearson']  # ['hed_bce']  # ['bsds_bce']  # ['bsds_bce']
    exp['optimizer'] = ['adam_w']  # , 'adam']
    exp['train_batch_size'] = [1]
    exp['val_batch_size'] = [1]
    exp['test_batch_size'] = [1]
    exp['epochs'] = [32]
    exp['all_results'] = True

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        # 'lr_flip_image_label',
        # 'rot_image_label',
        # 'ilsvrc12_normalize',
        'random_mask',
        'pascal_normalize',
        'image_to_bgr',
    ]]
    exp['val_augmentations'] = [[
        # 'bsds_mean',
        # 'bsds_normalize',
        'fixed_mask',
        'pascal_normalize',
        # 'ilsvrc12_normalize',
        'image_to_bgr',
    ]]
    exp['test_augmentations'] = exp['val_augmentations']
    return exp


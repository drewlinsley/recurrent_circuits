import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'cluttered_nist_3_ix2v2_45k',
    ]
    exp['val_dataset'] = [
        'cluttered_nist_3_ix2v2_45k',
    ]
    exp['model'] = [
        'resnet_18',
        'resnet_50',
        'resnet_152',
        'htd_lstm',
        'htd_gru',
    ]
    exp['validation_period'] = [2000]
    exp['validation_steps'] = [625]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [True]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1e-3, 1e-4, 1e-5, 1e-6]
    exp['loss_function'] = ['bce']
    exp['score_function'] = ['accuracy']
    exp['optimizer'] = ['adam']  # , 'adam']
    exp['train_batch_size'] = [32]
    exp['val_batch_size'] = [32]
    exp['epochs'] = [48]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'grayscale',
        'resize',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        # 'singleton',
        'zero_one'
    ]]
    exp['val_augmentations'] = [[
        'grayscale',
        'resize',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        # 'singleton',
        'zero_one'
    ]]
    return exp

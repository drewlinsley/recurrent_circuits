CUDA_VISIBLE_DEVICES=5 python run_job.py --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_09_06_12_12_49_585278/model_97700.ckpt-97700 --model=BSDS_vgg_gratings_simple --experiment=gratings_test --test --placeholders --out_dir=gratings_010 --no_db



CUDA_VISIBLE_DEVICES=5 python run_job.py --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_gratings_2019_09_06_19_10_10_192867/model_23400.ckpt-23400 --model=BSDS_vgg_gratings_simple --experiment=gratings_test --test --placeholders --out_dir=gratings_control_full --no_db


CUDA_VISIBLE_DEVICES=5 python run_job.py --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_horizontal_gratings_2019_09_06_22_39_40_587370/model_3700.ckpt-3700 --model=BSDS_vgg_gratings_horizontal --experiment=gratings_test --test --placeholders --out_dir=gratings_100_horizontal --no_db


CUDA_VISIBLE_DEVICES=5 python run_job.py --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_gratings_simple_untied_gratings_2019_09_06_22_45_52_666482/model_34900.ckpt-34900 --model=BSDS_vgg_gratings_simple_untied --experiment=gratings_test --test --placeholders --out_dir=gratings_100_untied --no_db


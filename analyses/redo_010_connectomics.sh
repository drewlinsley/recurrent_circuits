CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos_test --train=snemi_100 --val=snemi_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_09_23_21_54_32_660798/model_1200.ckpt-1200 --test --out_dir=snemi_005 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=berson_combos_test --train=berson_100 --val=berson_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_09_23_21_54_04_670946/model_500.ckpt-500 --test --out_dir=berson_005 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=snemi_combos_test --train=snemi_100 --val=snemi_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_09_23_21_54_28_198156/model_1800.ckpt-1800 --test --out_dir=snemi_005 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=berson_combos_test --train=berson_100 --val=berson_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_berson_combos_2019_09_23_21_54_22_374936/model_700.ckpt-700 --test --out_dir=berson_005 --no_db


# # 005

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_10_02_22_31_37_874997/model_250.ckpt-250 --test --out_dir=snemi_005 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_10_03_08_27_46_726349/model_500.ckpt-500 --test --out_dir=berson_005 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_10_02_22_31_26_849782/model_450.ckpt-450 --test --out_dir=snemi_005 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_berson_combos_2019_10_03_08_28_44_700220/model_200.ckpt-200 --test --out_dir=berson_005 --no_db

# # 010

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_10_03_15_00_18_641153/model_850.ckpt-850 --test --out_dir=snemi_010 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_10_03_08_28_50_272710/model_550.ckpt-550 --test --out_dir=berson_010 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_10_02_22_31_45_257860/model_750.ckpt-750 --test --out_dir=snemi_010 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_berson_combos_2019_10_03_15_01_57_633874/model_300.ckpt-300 --test --out_dir=berson_010 --no_db

# # 100

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_10_03_15_03_02_517799/model_1050.ckpt-1050 --test --out_dir=snemi_100 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_10_03_08_29_23_378103/model_600.ckpt-600 --test --out_dir=berson_100 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_10_04_12_49_20_822419/model_2250.ckpt-2250 --test --out_dir=snemi_100 --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_berson_combos_2019_10_03_08_29_02_967777/model_1550.ckpt-1550 --test --out_dir=berson_100 --no_db



# # 100

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos_augs --train=snemi_100_full --val=snemi_100_full --no_db

# CUDA_VISIBLE_DEVICES=1 python run_job.py --model=seung_unet_per_pixel --experiment=berson_combos_augs --train=berson_100_full --val=berson_100_full --no_db

# CUDA_VISIBLE_DEVICES=2 python run_job.py --model=refactored_v7 --experiment=snemi_combos_augs --train=snemi_100_full --val=snemi_100_full --no_db

# CUDA_VISIBLE_DEVICES=3 python run_job.py --model=refactored_v7 --experiment=berson_combos_augs --train=berson_100_full --val=berson_100_full --no_db


# 100 augs

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_augs_2020_02_03_18_06_51_085638/model_34650.ckpt-34650 --test --out_dir=snemi_100A --no_db

# CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_augs_2020_02_03_18_06_55_184209/model_9550.ckpt-9550 --test --out_dir=berson_100A --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=snemi_combos_test --train=snemi_100_full --val=snemi_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_augs_2020_02_05_17_33_46_609244/model_26750.ckpt-26750 --test --out_dir=snemi_100Av2 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=berson_combos_test --train=berson_100_full --val=berson_100_full --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_berson_combos_augs_2020_02_05_17_33_49_822053/model_11350.ckpt-11350 --test --out_dir=berson_100Av2 --no_db


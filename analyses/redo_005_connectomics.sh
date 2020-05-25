CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos_test --train=snemi_100 --val=snemi_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_snemi_combos_2019_09_23_11_54_45_887210/model_1100.ckpt-1100 --test --out_dir=snemi_005 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=seung_unet_per_pixel --experiment=berson_combos_test --train=berson_100 --val=berson_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/seung_unet_per_pixel_berson_combos_2019_09_22_22_40_23_113360/model_350.ckpt-350 --test --out_dir=berson_005 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=snemi_combos_test --train=snemi_100 --val=snemi_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_snemi_combos_2019_09_23_11_46_35_972481/model_1000.ckpt-1000 --test --out_dir=snemi_005 --no_db

CUDA_VISIBLE_DEVICES=0 python run_job.py --model=refactored_v7 --experiment=berson_combos_test --train=berson_100 --val=berson_100 --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/refactored_v7_berson_combos_2019_12_19_16_42_25_844469/model_300.ckpt-300 --test --out_dir=berson_005 --no_db


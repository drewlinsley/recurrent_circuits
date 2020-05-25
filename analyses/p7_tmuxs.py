# FFN
CUDA_VISIBLE_DEVICES=5 python train_wrapper_ts_1.py 12

# Multicue
drew@serrep7:/media/data_cifs/cluster_projects/bsds_ibm$ python run_job.py --model=BSDS_vgg_cheap_deepest_final_simple --experiment=multicue_edges_combos_flips_zooms_bdcnloss --no_db --placeholder_test=multicue_edges_test --num_gpus=8

# Seung on snemi 100
serrep7:/media/data_cifs/cluster_projects/refactor_gammanet$ CUDA_VISIBLE_DEVICES=1 python run_job.py --model=seung_unet_per_pixel --experiment=snemi_combos --train=snemi_100_full --val=snemi_100_full --no_db

# GN on snemi 100
CUDA_VISIBLE_DEVICES=3 python run_job.py --model=refactored_v7 --experiment=snemi_combos --train=snemi_100 --val=snemi_100 --no_db

# Disentangling paper experiment
/media/data_cifs/cluster_projects/refactor_gammanet$ CUDA_VISIBLE_DEVICES=3 python run_job.py --experiment=pathfinder_14_60k_no_db --no_db --model=htd_fgru_shallow

# Run i3D training
drew@serrep7:/media/data_cifs/cluster_projects/TPU_Projects$ bash run_train_i3d_local.sh

# Reset berson DB
drew@serrep7:/media/data_cifs/cluster_projects/ffn_membrane_v2$ python db_tools.py --reset_coordinates --reset_config --reset_priority --priority_list=db/test_priorities.csv

# Run KUBER
drew@serrep7:~/Documents$ python3.7 -m wkcuber.convert_knossos --layer_name color /media/data_cifs/connectomics/mag1 /media/data/cubed_mag1^C

# Train on gratings
drew@serrep7:/media/data_cifs/cluster_projects/undo_bias$ CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=gratings --model=BSDS_vgg_gratings_simple --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_09_06_15_32_59_371313/model_980.ckpt-980
drew@serrep7:/media/data_cifs/cluster_projects/refactor_gammanet$ CUDA_VISIBLE_DEVICES=1 python run_job.py --experiment=gratings --model=BSDS_vgg_gratings_simple --no_db --ckpt=/media/data_cifs/cluttered_nist_experiments/checkpoints/BSDS_vgg_cheap_deepest_final_simple_BSDS500_combos_100_no_aux_2019_09_06_15_32_59_371313/model_980.ckpt-980

# BSDS training
drew@serrep7:/media/data_cifs/cluster_projects/bsds_ibm$ CUDA_VISIBLE_DEVICES=2 python run_job.py --experiment=BSDS500_combos_100_no_aux --no_db --model=BSDS_vgg_cheap_deepest_final_simple_combined_untied --train=BSDS500_100_jk --val=BSDS500_100_jk --placeholder_test=BSDS500_test_padded



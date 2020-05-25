#!/bin/bash

COUNTER=0
while [ $COUNTER -lt 10 ]; do
	echo Step $COUNTER
	CUDA_VISIBLE_DEVICES=1 python run_job.py --experiment=v2_synth_connectomics_baseline_combos --train=v2_synth_connectomics_baseline_5 --val=v2_synth_connectomics_baseline_20 --model=seung_unet_per_pixel_adabn --no_db
	CUDA_VISIBLE_DEVICES=1 python run_job.py --experiment=snemi_combos --train=snemi_5 --val=snemi_1 --model=seung_unet_per_pixel_adabn --no_db
	CUDA_VISIBLE_DEVICES=1 python run_job.py --experiment=snemi_combos --train=snemi_20 --val=snemi_1 --model=seung_unet_per_pixel_adabn --no_db
	let COUNTER=COUNTER+1
done


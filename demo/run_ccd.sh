#!/bin/bash

python main.py \
    --study tcga_blca \
    --task survival \
    --modality ccd_survpath \
    --type_of_path combine \
    --n_classes 4 \
    --max_epochs 20 \
    --lr 0.0005 \
    --reg 0.0001 \
    --alpha_ccd 0.1 \
    --beta_ccd 0.5 \
    --batch_size 1 \
    --label_col survival_months_dss \
    --data_root_dir /data/TCGA/BLCA/features/pt_files/ \
    --label_file datasets_csv/metadata/tcga_blca.csv \
    --omics_dir datasets_csv/raw_rna_data/combine/blca \
    --results_dir ccd_blca_results \
    --which_splits 5foldcv \
    --seed 42 \
    --wsi_projection_dim 512 \
    --num_patches 4090 \
    --use_counterfactual \
    --bag_loss nll_surv

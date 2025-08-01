#!/bin/bash

for inter in no_network 1D-CNN GRU 1D-CNN_GRU
do
	for pooler in mean max cls_token convolutional_attention
	do
		for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
		do
			python ../../DeepCOI-dev/predict.py \
				--config_path ../../DeepCOI-dev/config/DeepCOI-t6-k4/ \
				--model ../Table2/models/${inter}/${pooler}.${phylum}.pt \
				--seq ../../data/${phylum}.test.fasta \
				--batch_size 4 \
				--output ${inter}/${pooler}.${phylum}.test.npy \
				--save_probs \
				--mcm
		done
	done
done
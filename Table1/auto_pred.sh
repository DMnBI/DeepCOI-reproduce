#!/bin/bash

for model in onehot random end_to_end DeepCOI
do
	for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
	do
		python ../DeepCOI-dev/predict.py \
			--config_path ../DeepCOI-dev/config/DeepCOI-t6-k4/ \
			--model models/${model}.${phylum}.pt \
			--seq ../data/${phylum}.test.fasta \
			--batch_size 4 \
			--output ${model}.${phylum}.test.npy \
			--save_probs \
			--mcm
	done
done
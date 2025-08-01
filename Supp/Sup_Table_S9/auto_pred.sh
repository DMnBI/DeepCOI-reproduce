#!/bin/bash

for trial in 0 1 2 3 4
do
	for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
	do
		python ../../DeepCOI-dev/predict.py \
			--config_path ../../DeepCOI-dev/config/DeepCOI-t6-k4/ \
			--model models/DeepCOI-${phylum}-${trial}.pt \
			--seq ../../data/${phylum}.test.fasta \
			--batch_size 4 \
			--output DeepCOI-${phylum}-${trial}.test.npy \
			--save_probs \
			--mcm
	done
done
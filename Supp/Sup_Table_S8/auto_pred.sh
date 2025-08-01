#!/bin/bash

for width in 5 7 9 11 13
do
	for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
	do
		python ../../DeepCOI-dev/predict.py \
			--config_path ../../DeepCOI-dev/config/DeepCOI-t6-k4/ \
			--model models/DeepCOI-${phylum}-cnn${width}.pt \
			--seq ../../data/${phylum}.test.fasta \
			--batch_size 4 \
			--output ${phylum}-cnn${width}.test.npy \
			--save_probs \
			--mcm
	done
done
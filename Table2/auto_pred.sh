#!/bin/bash

for inter in no_network GRU 1D-CNN_GRU
do
	for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
	do
		python ../DeepCOI-dev/predict.py \
			--config_path ../DeepCOI-dev/config/DeepCOI-t6-k4/ \
			--model models/${inter}/convolutional_attention.${phylum}.pt \
			--seq ../data/${phylum}.test.fasta \
			--batch_size 4 \
			--output ${inter}.convolutional_attention.${phylum}.test.npy \
			--save_probs \
			--mcm
	done
done

for inter in 1D-CNN
do
	for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
	do
		python ../DeepCOI-dev/predict.py \
			--config_path ../DeepCOI-dev/config/DeepCOI-t6-k4/ \
			--model models/${inter}/max.${phylum}.pt \
			--seq ../data/${phylum}.test.fasta \
			--batch_size 4 \
			--output ${inter}.max.${phylum}.test.npy \
			--save_probs \
			--mcm
	done
done
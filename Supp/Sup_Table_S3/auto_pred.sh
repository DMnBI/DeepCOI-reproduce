#!/bin/bash

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../../DeepCOI-dev/predict.py \
		--config_path ../../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/mcm-${phylum}.pt \
		--seq ../../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output w_mcm.w_mcm.${phylum}.test.npy \
		--save_probs \
		--mcm
done

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../../DeepCOI-dev/predict.py \
		--config_path ../../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/mcm-${phylum}.pt \
		--seq ../../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output w_mcm.wo_mcm.${phylum}.test.npy \
		--save_probs
done

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../../DeepCOI-dev/predict.py \
		--config_path ../../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/DeepCOI-${phylum}.pt \
		--seq ../../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output wo_mcm.wo_mcm.${phylum}.test.npy \
		--save_probs
done

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../../DeepCOI-dev/predict.py \
		--config_path ../../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/DeepCOI-${phylum}.pt \
		--seq ../../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output wo_mcm.w_mcm.${phylum}.test.npy \
		--save_probs \
		--mcm
done
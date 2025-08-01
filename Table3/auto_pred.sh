#!/bin/bash

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../DeepCOI-dev/predict.py \
		--config_path ../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/base-${phylum}.pt \
		--seq ../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output base.${phylum}.test.npy \
		--save_probs
done

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../DeepCOI-dev/predict.py \
		--config_path ../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/base-${phylum}.pt \
		--seq ../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output wo_wBCE.${phylum}.test.npy \
		--save_probs \
		--mcm
done

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../DeepCOI-dev/predict.py \
		--config_path ../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/DeepCOI-${phylum}.pt \
		--seq ../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output wo_MCM.${phylum}.test.npy \
		--save_probs
done

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes
do
	python ../DeepCOI-dev/predict.py \
		--config_path ../DeepCOI-dev/config/DeepCOI-t6-k4/ \
		--model models/DeepCOI-${phylum}.pt \
		--seq ../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output DeepCOI.${phylum}.test.npy \
		--save_probs \
		--mcm
done
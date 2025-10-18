#!/bin/bash

DEEPCOI="/path/to/DEEPCOI"

for phylum in Annelida Arthropoda Chordata Cnidaria Echinodermata Mollusca Nematoda Platyhelminthes outgroups
do
	python ${DEEPCOI}/src/deepcoi predict \
		--config_path ${DEEPCOI}/config/DeepCOI-t6-k4/ \
		--model ${DEEPCOI}/models/fine-tuned/DeepCOI-phylum.pt \
		--seq ../../data/${phylum}.test.fasta \
		--batch_size 4 \
		--output DeepCOI-phylum.${phylum}.npy \
		--save_probs \
		--phylum
done
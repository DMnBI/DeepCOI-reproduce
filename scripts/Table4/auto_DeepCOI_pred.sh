#!/bin/bash

DEEPCOI="/path/to/DeepCOI"

for i in 1 2 3 4
do
	python ${DEEPCOI}/src/deepcoi predict \
		--config_path ${DEEPCOI}/config/DeepCOI-t6-k4/ \
		--model ../models/Table4/DeepCOI-Arthropoda_woDS-PBBC.pt \
		--seq ../data/realworld/DS-PBBC${i}.fasta \
		--batch_size 4 \
		--output DeepCOI/raw/DS-PBBC${i}.pred.txt \
		--mcm
done

for i in 1 2 3 4
do
	python ${DEEPCOI}/src/deepcoi predict \
		--config_path ${DEEPCOI}/config/DeepCOI-t6-k4/ \
		--model ../models/Table4/DeepCOI-Arthropoda_woDS-PBBC.pt \
		--seq ../data/realworld/DS-PBBC${i}.fasta \
		--batch_size 4 \
		--output DeepCOI/raw/DS-PBBC${i}.pred.no_mcm.txt
done
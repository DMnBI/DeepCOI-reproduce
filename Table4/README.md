# Table4
This directory is related to the Table4

## Related models
**Zenodo:** models\_Table4.tar.gz  
models/  
├ DeepCOI-Arthropoda_woDS-PBBC.pt   

## Related data
../data/  
├ realworld/  
│ ├ labels/  
│ ├ DS-PBBC{1..4}.fasta  
│ ├ rdp_woDS-PBBC/  

## How to reproduce DeepCOI results
Before run the following script, please install DeepCOI properly and download all the related fine-tuned models. 

Excute following scripts within the `Table4` directory  

### 1. Run DeepCOI
You should edit the following shell script as replacing DeepCOI installed path.

```
bash ../scripts/Table4/auto_DeepCOI_pred.sh
```
The above script will run `DeepCOI` for `../data/realworld/DS-PBBC{1..4}.fasta` files against fine-tuned models.  

### 2. Summarize DeepCOI results
```
python ../scripts/Table4/summary_DeepCOI.py
```
The above script will generate `DeepCOI/DeepCOI.summary.csv` and `DeepCOI/DeepCOI.no_mcm.summary.csv`

## How to reproduce BLAST results
`BLAST 2.16.0+` was used for this study.  

Excute following scripts within the `Table4/` directory  

### 1. Build BLAST db without sample sequences
```
python ../scripts/Table4/makeblastdb.py
```
The above script will generate `Arthropoda_woDS-PBBC.train.fasta` by filtering sequences according to their sequence ID. This file is the sample file that was used for training DeepCOI and RDP classifier.  
Then, makeblastdb command is also automatically run within the above script.  

### 2. BLAST search
```
python ../scripts/Table4/auto_BLAST_search.py
```
The above script will run `blastn` for `../data/realword/DS-PBBC{1..4}.fasta` files against previously built DBs  

### 3. Summarize BLAST results
```
python ../scripts/Table4/summary_BLAST.py
```
The above script will generate `BLAST/DS-PBBC.blast.csv`  


## How to reproduce RDP results
`RDP classifier 2.14` was used for this study. 

Excute following scripts within the `Table4/` directory  

### 1. RDP search
You should first install `rdp_classifier_2.14` in `DeepCOI-reproduce/data/RDP/` directory or replace `classifier` path in the following script by your installed path.

```
python ../scripts/Table4/auto_RDP_search.py
```
The above script will run `rdp classifier` for `../data/realword/DS-PBBC{1..4}.fasta` files against pretrained RDP classifiers.

### 2. Summarize BLAST results
```
python ../scripts/Table4/summary_RDP.py
```
The above script will generate `RDP/DS-PBBC.rdp.csv`  
# DeepCOI-reproduce
A repository for reproducing Figures and Tables of DeepCOI paper

To reproduce the whole results published in our paper, please make sure download and place all data properly  
 
The following guidance was written in assumption that you have already placed all related data in `DeepCOI-reproduce/data/` directory  

## Repository structure
We highly recommend placing the downloaded file in the proper directory structure.  

**DeepCOI-reproduce/**  
Figure3/  
├ BLAST/  
├ ├ dbs/  
├ ├ m6/  
├ ├ BLAST.scores.csv  
├ ├ BLAST.scores.included.csv  
├ ├ BLAST.scores.excluded.csv  
├ RDP/  
├ ├ preds/  
├ ├ RDP.scores.csv  
├ ├ RDP.scores.included.csv  
├ ├ RDP.scores.excluded.csv  
├ DeepCOI/  
├ ├ preds/  
├ ├ DeepCOI.scores.csv  
├ ├ DeepCOI-included.scores.csv  
├ ├ DeepCOI-excluded.scores.csv  
├ excluded.csv  


## How to reproduce DeepCOI results
Before run the following script, please install DeepCOI properly and download all the related fine-tuned models. 

Excute following scripts within the `Figure3/DeepCOI` directory  

### 1. Run DeepCOI
You should edit the following shell script as replacing DeepCOI installed path.

```
bash ../../scripts/Figure3/auto_DeepCOI_pred.sh
```
The above script will run `DeepCOI` for `../../data/*.test.fasta` files against fine-tuned models.

### 2. Measure DeepCOI performance
```
python ../../scripts/Figure3/get_DeepCOI_score.py
```
The above script will generate `DeepCOI.scores.csv`, `DeepCOI-included.scores.csv`, `DeepCOI-excluded.scores.csv` automatically.  
It will take several hours to several days according to your computational power.  
> Arthropoda and Chordata especially take long time to compute due to their data size.

## How to reproduce BLAST results
`BLAST 2.16.0+` was used for this study.  

Excute following scripts within the `Figure3/BLAST` directory  

### 1. Build BLASTn databases
```
python ../../scripts/Figure3/auto_build_dbs.py
```
The above script will run `makeblastdb` for `../../data/*.train.fasta` automatically.  

### 2. BLAST search
```
python ../../scripts/Figure3/auto_BLAST_search.py
```
The above script will run `blastn` for `../../data/*.test.fasta` files against previously built DBs  

### 3. Measure BLAST performance
```
python ../../scripts/Figure3/get_BLAST_score.py
```
The above script will generate `BLAST.scores.csv`, `BLAST.scores.included.csv`, `BLAST.scores.excluded.csv` automatically.  
It will take several hours to several days according to your computational power.  
> Arthropoda and Chordata especially take long time to compute due to their data size.

## How to reproduce RDP results
`RDP classifier 2.14` was used for this study. 

Excute following scripts within the `Figure3/RDP` directory  

### 1. RDP search
You should first install `rdp_classifier_2.14` in `DeepCOI-reproduce/data/RDP/` directory or replace `classifier` path in the following script by your installed path.

```
python ../../scripts/Figure3/auto_RDP_search.py
```
The above script will run `rdp classifier` for `../../data/*.test.fasta` files against pretrained RDP classifiers.

### 2. Measure RDP performance
```
python ../../scripts/Figure3/get_RDP_score.py
```
The above script will generate `RDP.scores.csv`, `RDP.scores.included.csv`, `RDP.scores.excluded.csv` automatically.  
It will take several hours to several days according to your computational power.  
> Arthropoda and Chordata especially take long time to compute due to their data size.  

## Concatenate results for excluded species
Excute following scripts within the `Figure3/` directory  

```
python ../scripts/Figure3/concat_excluded.py
```
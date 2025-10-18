# DeepCOI-reproduce
A repository for reproducing Figures and Tables of DeepCOI paper

To reproduce the whole results published in our paper, please make sure download and place all data properly  
 
The following guidance was written in assumption that you have already placed all related data in `DeepCOI-reproduce/data/` directory  

## Repository structure
We highly recommend placing the downloaded file in the proper directory structure.  

**DeepCOI-reproduce/**  
Figure2/  
├ RDP/  
├ ├ {phylums}.rdp.test.txt   
├ DeepCOI/  
├ ├ DeepCOI-phylum.{phylums}.npy   


## How to reproduce DeepCOI results
Before run the following script, please install DeepCOI properly and download all the related fine-tuned models. 

Excute following scripts within the `Figure2/DeepCOI` directory  

### 1. Run DeepCOI
You should edit the following shell script as replacing DeepCOI installed path.

```
bash ../../scripts/Figure2/auto_DeepCOI_pred.sh
```
The above script will run `DeepCOI` for `../../data/*.test.fasta` files against fine-tuned models for the phylum level.

## How to reproduce RDP results
`RDP classifier 2.14` was used for this study. 

Excute following scripts within the `Figure2/RDP` directory  

### 1. RDP search
You should first install `rdp_classifier_2.14` in `DeepCOI-reproduce/data/RDP/` directory or replace `classifier` path in the following script by your installed path.

```
python ../../scripts/Figure2/auto_RDP_search.py
```
The above script will run `rdp classifier` for `../../data/*.test.fasta` files against pretrained RDP classifiers for the phylum level.
# DeepCOI-reproduce
A repository for reproducing Figures and Tables of DeepCOI paper

**models:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16593248.svg)](https://doi.org/10.5281/zenodo.16593248)  
**data:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16593030.svg)](https://doi.org/10.5281/zenodo.16593030)

## Repository structure
We highly recommend placing the downloaded file in the proper directory structure.  

**DeepCOI-reproduce/**  
├ data/  
├ ├ meta/ (files in meta\_data.tar.gz)  
├ ├ realworld/ (files in realworld.tar.gz)  
├ ├ test\_labels/ (files in test\_labeles.tar.gz)  
├ ├ (files in training\_data.tar.gz)  
├ Table1/  
├ ├ models/ (models in models\_Table1.tar.gz)  
├ ├ \*.test.npy  
├ Table2/  
├ ├ models/  
├ ├ ├ no\_network (models in models\_T2\_no_network.tar.gz)  
├ ├ ├ 1D-CNN (models in models\_T2\_1D-CNN.tar.gz)  
├ ├ ├ GRU (models in models\_T2\_GRU.tar.gz)  
├ ├ ├ 1D-CNN_GRU (models in models\_T2\_1D-CNN\_GRU.tar.gz)  
├ ├ no\_network/  
├ ├ ├ \*.test.npy  
├ ├ 1D-CNN/  
├ ├ ├ \*.test.npy  
├ ├ GRU/  
├ ├ ├ \*.test.npy  
├ ├ 1D-CNN\_GRU/  
├ ├ ├ \*.test.npy  
├ Table3/  
├ ├ models/ (models in models\_Table3.tar.gz)  
├ ├ \*.test.npy  
├ Table4/  
├ ├ models/ (models in models\_Table4.tar.gz)  
├ ├ \*.test.npy  
├ Figure2/  
├ ├ DeepCOI/ (files in DeepCOI.tar.gz)  
├ ├ RDP/ (files in RDP.tar.gz)  
├ Figure3/  
├ ├ BLAST/  
├ ├ RDP/  
├ ├ DeepCOI/  
├ ├ excluded.csv  
├ Figure4/  
├ ├ running_times.csv  
├ Figure5/  
├ scripts/  (scripts used to parse data for tables)  
├ notebooks/  
├ Supp/  
├ ├ ...  

## Download raw records from BOLD systems
We collected the raw records from the BOLD systems v4 [https://v4.boldsystems.org](https://v4.boldsystems.org) on Aug 22, 2022. If you need the raw records, you can download using the following script or instruction.  

> [!NOTE]  
> BOLD systems has now been updated to v5.  
> The following script/instruction can download records currently available in BOLD systems

### 1. Using download script

Run the following script in `DeepCOI-reproduce/data/` directory.

```
python ../scripts/download_raw.py {phylum what you want to download}

[example]
python ../scripts/download_raw.py Annelida
```
This script will generate `{phylum}.raw.tsv` file.

### 2. From data portal
You can visit data portal [https://portal.boldsystems.org/](https://portal.boldsystems.org/). To download raw records, you are going to search target phylum (or target specimem) and click the `tsv` button. `results.tsv` file will be downloaded into your Download directory.  

> [!NOTE]  
> This instruction is no longer allowed for large datasets (such as Arthropoda) in BOLDsystems v5.
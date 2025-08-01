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
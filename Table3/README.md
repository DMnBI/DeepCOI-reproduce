# Table3
This directory is related to the Table3

## Related models
**Zenodo:** models\_Table3.tar.gz  
models/  
├ base-{eight phyla}.pt   
├ DeepCOI-{eight phyla}.pt  

> [!NOTE]  
> For DeepCOI models, you have to copy or soft link "../Supp/Sup_Fig_S2/t6-320.{eight phyla}.pt"

## run predictions
run `auto_pred.sh` for conducting predictions. This script will generate  {base,wo_MCM,wo_wBCE,DeepCOI}.{eight phyla}.test.npy files

To generate a whole table, please edit 25,26th lines of `../scripts/Table3.py`

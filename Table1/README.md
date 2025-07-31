# Table1
This directory is related to the Table1

## Related models
**Zenodo:** models_Table1.tar.gz  
models/  
├ onehot\-{eight phyla}.pt  
├ random\-{eight phyla}.pt  
├ end\_to\_end\-{eight phyla}.pt  
├ DeepCOI\-{eight phyla}.pt  

> [!NOTE]  
> For DeepCOI models, you have to copy or soft link "../Supp/Sup_Fig_S2/t6-320.{eight phyla}.pt"

## run predictions
run `auto_pred.sh` for conducting predictions. This script will generate {onehot,random,end\_to\_end,DeepCOI}.{eight phyla}.test.npy files

To generate a whole table, please edit 25,26th lines of `../scripts/Table1.py`

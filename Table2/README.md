# Table2
This directory is related to the Table2

## Related models
**Zenodo:**  
models\_T2\_no\_network.tar.gz  
models\_T2\_1DCNN.tar.gz  
models\_T2\_GRU.tar.gz  
models\_T2\_1DCNN_GRU.tar.gz
 
models/  
├ no\_network  
│├ mean.{eight phyla}.pt  
│├ max.{eight phyla}.pt  
│├ cls\_toekn.{eight phyla}.pt  
│├ convolutional\_attention.{eight phyla}.pt  
├ 1D-CNN  
│├ mean.{eight phyla}.pt  
│├ max.{eight phyla}.pt  
│├ cls\_toekn.{eight phyla}.pt  
│├ convolutional\_attention.{eight phyla}.pt  
├ GRU  
│├ mean.{eight phyla}.pt  
│├ max.{eight phyla}.pt  
│├ cls\_toekn.{eight phyla}.pt  
│├ convolutional\_attention.{eight phyla}.pt  
├ 1D-CNN_GRU  
│├ mean.{eight phyla}.pt  
│├ max.{eight phyla}.pt  
│├ cls\_toekn.{eight phyla}.pt  
│├ convolutional\_attention.{eight phyla}.pt    

> [!NOTE]  
> For 1D-CNN/max... models, you have to copy or soft link "../Supp/Sup_Fig_S2/t6-320.{eight phyla}.pt"

## run predictions
run `auto_pred.sh` for conducting predictions. This script will generate  {no_network,1D-CNN,GRU,1D-CNN\_GRU}/{mean,max,cls\_token,convolutional\_attention}.{eight phyla}.test.npy files

To generate a whole table, please edit 25,26th lines of `../scripts/Table2.py`

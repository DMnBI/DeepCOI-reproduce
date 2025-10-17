from metrics import *

import numpy as np
import pandas as pd
import itertools as it
import pickle as pkl

from datetime import datetime
import pytz

def log(msg):
	ctime = datetime.now(pytz.timezone('Asia/Seoul'))
	ts = ctime.strftime("%Y-%m-%d %H:%M:%S")

	print(f"[{ts}] {msg}", flush=True)

def make_pred_vec(preds, names, taxa_info):
    y_pred = []
    for rank in RANKS:
        spos, epos = taxa_info[rank]
        t_names = names[spos:epos]
        idx = np.where(t_names == f"{rank[0]}__{preds[rank][0]}")
        prob = preds[rank][1]

        t_pred = np.full_like(t_names, (1-prob)/(t_names.shape[0]-1), dtype=np.double)
        t_pred[idx] = prob

        y_pred.append(t_pred)
        
    return np.concatenate(y_pred)
    
def parse_id2label(file_name):
    id2label = {}
    with open(file_name, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            items = line.strip().split(' |')
            id2label[items[0][1:]] = items[-1]
            
    return id2label
    
def get_rdp_trues(labels, names, DAG):
    tmp = []
    for label in labels:
        try:
            species = f"s__{label}"
            idx = np.where(names == species)[0][0]
        except:
            genus = f"g__{label}"
            idx = np.where(names == genus)[0][0]
            
        label = DAG[:, idx]
        tmp.append(label)
        
    return np.array(tmp)
    
def parse_rdp(file_name, meta, id2label, targets=None):
    names = meta['names']
    DAG = meta['DAG'][()].toarray()
    taxa_info = get_taxa_info(names)
    
    with open(file_name, 'r') as f:
        labels = []
        y_preds = []
        for line in f:
            items = line.strip().split('\t')
            sid = items[0]
            if targets is not None and sid not in targets.values:
                continue
            preds = {items[idx+1]:(items[idx], float(items[idx+2])) for idx in range(2, len(items), 3)}

            labels.append(id2label[sid])
            y_preds.append(make_pred_vec(preds, names, taxa_info))

        y_trues = get_rdp_trues(labels, names, DAG)
        y_preds = np.stack(y_preds)
        
    return y_trues, y_preds

def melt(table, metric):
    df = table.unstack().reset_index()
    df.columns = ['rank', 'phylum', metric]
    df['pivot'] = [f"{row['phylum']}.{row['rank']}" for _, row in df.iterrows()]

    return df[['pivot', metric]]

def make_dict(PHYLA):
	return {phylum:dict() for phylum in PHYLA}

def report_scores(subset=None):
    RANKS = ['class', 'order', 'family', 'genus', 'species']
    PHYLA = ['Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']

    fpr, tpr, roc_auc = make_dict(RANKS), make_dict(RANKS), make_dict(RANKS)
    precision, recall, pr_auc = make_dict(RANKS), make_dict(RANKS), make_dict(RANKS)

    for phylum in tqdm(PHYLA):
        log(f"{phylum} START.")
        meta = np.load(f"../../data/meta/{phylum}.meta.npz", allow_pickle=True)
        names = meta['names']
        taxa_info = get_taxa_info(names)

        labels = pd.read_csv(f"../../data/testset_labels/{phylum}.test.labels.txt", sep=',')
        if subset == 'included':
            targets = labels.query("species == 'included'")['sid']
        elif subset == 'excluded':
            targets = labels.query("species == 'excluded'")['sid']
        else:
            targets = None
        
        id2label = parse_id2label(f"../../data/{phylum}.test.fasta")
        y_true, y_pred = parse_rdp(f"preds/{phylum}.rdp.test.txt", meta, id2label, targets=targets)
        
        for rank in RANKS:
            spos, epos = taxa_info[rank]
            sub_true = y_true[:, spos:epos]
            sub_pred = y_pred[:, spos:epos]
            names = meta['names'][spos:epos]
            
            # ROC curve
            x, y, z, (a, b) = prepare(sub_true, sub_pred, names, metric='roc', return_th=True)
            fpr[rank][phylum], tpr[rank][phylum], roc_auc[rank][phylum] = x, y, z
    #        th[rank][phylum], th_fpr[rank][phylum] = a, b
            
            # PR curve
            x, y, z = prepare(sub_true, sub_pred, names, metric='pr')
            precision[rank][phylum], recall[rank][phylum], pr_auc[rank][phylum] = x, y, z
            
    rdp_roc_table = pd.DataFrame({rank:{phylum:roc_auc[rank][phylum]['macro'] for phylum in PHYLA} for rank in RANKS})
    rdp_pr_table = pd.DataFrame({rank:{phylum:pr_auc[rank][phylum]['macro'] for phylum in PHYLA} for rank in RANKS})

    rdp_roc_table = melt(rdp_roc_table, 'AUROC')
    rdp_pr_table = melt(rdp_pr_table, 'AUPR')

    df = pd.merge(left=rdp_roc_table, right=rdp_pr_table, left_on='pivot', right_on='pivot')
    df['phylum'] = df['pivot'].apply(lambda x: x.split('.')[0])
    df['rank'] = df['pivot'].apply(lambda x: x.split('.')[1])
    df = df[['phylum', 'rank', 'AUROC', 'AUPR']]

    output = 'test_RDP.scores.csv'
    if targets is not None:
        output = f'RDP.scores.{subset}.csv'
    df.to_csv(output, index=False)

report_scores(subset=None)
report_scores(subset='included')
report_scores(subset='excluded')
from typing import Union, Dict, Callable
from pathlib import Path
import itertools as it
import numpy as np
import scipy.sparse as ssp

import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import (
	EsmModel,
)

from torch.optim import Adam
from .lr_scheduler import InverseSqrtScheduler

from .metrics import accuracy, auroc
from .pooler import (
	MeanPooling,
	MaxPooling,
	CLRPooling,
	LightAttention
)
from .modeling_utils import (
	OneHotEncoding,
	SimpleMLP,
	CnnGru,
	Cnn,
	Gru
)


class DeepCOIEmbedding(nn.Module):
	def __init__(self,
		esm: Union[str, Path],
		ae: Union[str, Path] = None,
		vae: Union[str, Path] = None,
		**kwargs):
		super().__init__()

		self.is_ae = False
		self.is_vae = False

		self.esm = torch.load(esm)
		if ae is not None:
			self.ae = torch.load(ae)
			self.is_ae = True
		if vae is not None:
			self.vae = torch.load(vae)
			self.is_vae = True

		self.embed_size = self.esm.config.hidden_size

	def forward(self, input_ids, attention_mask=None):
		if attention_mask is None:
			attention_mask = input_ids.ne(self.esm.config.pad_token_id).float()

		embed_tokens = self.esm(
			input_ids,
			attention_mask=attention_mask
		)
		
		sequence_output = embed_tokens[0]
		return sequence_output


class DeepCOI(nn.Module):
	def __init__(self,
		meta: Union[str, Path],
		freeze_embedding: bool = False,
		cnn_width: int = 5,
		la_width: int = 9,
		middle_layer: str = 'Cnn',
		pooler: str = 'la',
		vocab_size: int = None,
		embedding: str = 'esm',
		mcm: bool = True,
		**kwargs):
		super().__init__()

		meta = np.load(meta, allow_pickle=True)
		self.names = meta['names']
		self.CM = torch.tensor(meta['DAG'][()].toarray())
		self.mcm = mcm

		self.embedding = None
		if embedding == 'esm':
			self.embedding = DeepCOIEmbedding(**kwargs)
		elif embedding == 'one-hot':
			self.embedding = OneHotEncoding(vocab_size)

		self.middle_layer=None
		if middle_layer == 'CnnGru':
			self.middle_layer = CnnGru(
				self.embedding.embed_size,
				self.embedding.embed_size,
				cnn_width=cnn_width
			)
		elif middle_layer == 'Cnn':
			self.middle_layer = Cnn(
				self.embedding.embed_size,
				self.embedding.embed_size,
				cnn_width=cnn_width
			)
		elif middle_layer == 'Gru':
			self.middle_layer = Gru(
				self.embedding.embed_size,
				self.embedding.embed_size
			)

		self.pooler = None
		if pooler == 'mean':
			self.pooler = MeanPooling()
		elif pooler == 'max':
			self.pooler = MaxPooling()
		elif pooler == 'clr':
			self.pooler = CLRPooling(self.embedding.embed_size)
		elif pooler == 'la':
			self.pooler = LightAttention(self.embedding.embed_size, width=la_width)
		elif pooler in ['concat', 'ensemble']:
			self.pooler = nn.ModuleList([
				MaxPooling(),
				LightAttention(self.embedding.embed_size, width=la_width)
			])

		dim1 = self.embedding.embed_size
		dim2 = self.embedding.embed_size // 2
		self.ensemble = False
		if pooler == 'concat':
			dim1 *= 2
			dim2 *= 2

			self.classifier = SimpleMLP(
				dim1, 
				dim2, 
				self.names.shape[0]
			)
		elif pooler == 'ensemble':
			self.ensemble = True
			self.classifier = nn.ModuleList([
				SimpleMLP(dim1, dim2, self.names.shape[0]),
				SimpleMLP(dim1, dim2, self.names.shape[0])
			])
		else:
			self.classifier = SimpleMLP(dim1, dim2, self.names.shape[0])

		if freeze_embedding:
			self.embedding.requires_grad_(False)

	def forward(self, input_ids, attention_mask=None, labels=None):
		if isinstance(self.embedding, DeepCOIEmbedding):
			embed_tokens = self.embedding(input_ids, attention_mask)
		else:
			embed_tokens = self.embedding(input_ids).float()

		if self.middle_layer is not None:
			embed_tokens = self.middle_layer(embed_tokens)
		if isinstance(self.pooler, nn.ModuleList) and not self.ensemble:
			p1 = self.pooler[0](embed_tokens)
			p2 = self.pooler[1](embed_tokens)
			pooled_output = torch.cat([p1, p2], dim=1)

			output = self.classifier(pooled_output).sigmoid()
		elif self.ensemble:
			p1 = self.pooler[0](embed_tokens)
			p2 = self.pooler[1](embed_tokens)

			o1 = self.classifier[0](p1).sigmoid()
			o2 = self.classifier[1](p2).sigmoid()

			output = torch.max(torch.stack([o1, o2]), dim=0).values
		else:
			pooled_output = self.pooler(embed_tokens)
			output = self.classifier(pooled_output).sigmoid()

		if self.mcm:
			_cm = self.CM.unsqueeze(0).to(input_ids.device)
			output = output.unsqueeze(1)
			if self.training:
				a = (1 - labels) * torch.max(output * _cm, dim=-1).values
				b = labels * torch.max(output * (_cm * labels.unsqueeze(1)), dim=-1).values
				output = a + b
			else:
				output = torch.max(output * _cm, dim=-1).values

		return output.float()

def get_taxa_info(names):
	RANKS = ['phylum', 'class', 'order', 'family', 'genus', 'species'][1:]
	num_taxa = {
		rank: np.char.startswith(names, f"{rank[0]}__").sum()
		for rank in RANKS
	}
	end_pos = list(it.accumulate(num_taxa.values()))
	start_pos = [0] + end_pos[:-1]

	taxa_info = {}
	for rank, spos, epos in zip(RANKS, start_pos, end_pos):
		taxa_info[rank[0]] = (spos, epos)

	return taxa_info


class DeepCOITraining(pl.LightningModule):
	def __init__(self,
		esm: Union[str, Path], 
		meta: Union[str, Path],
		embedding: str = 'esm',
		vocab_size: int = None,
		freeze_embedding: bool = False,
		cnn_width: int = 5,
		la_width: int = 9,
		middle_layer: str = 'Cnn',
		pooler: str = 'la',
		learning_rate: float = 5e-4, 
		adam_beta1: float = 0.9, 
		adam_beta2: float = 0.999, 
		adam_epsilon: float = 1e-6,
		mcm: bool = True,
		weighting_parents: bool = False,
		weighting_species: bool = False,
		lr_schedule: bool = False,
		warmup_steps: int = 1):
		super().__init__()
		self.save_hyperparameters()

		self.model = DeepCOI(
			esm = esm,
			meta = meta,
			embedding = embedding,
			vocab_size = vocab_size,
			freeze_embedding = freeze_embedding,
			cnn_width = cnn_width,
			la_width = la_width,
			middle_layer = middle_layer,
			pooler = pooler,
			mcm = mcm,
		)

		self.weights = torch.ones(self.model.names.shape[0])
		taxa_info = get_taxa_info(self.model.names)
		if weighting_parents:
			weights = []
			for i, name in enumerate(self.model.names):
				if name[0] == 's':
					weight = 1
				else:
					tinfo = taxa_info[name[0]]
					n_child = self.model.CM[i][tinfo[1]:].sum()
					weight = np.sqrt(n_child.item())
				weights.append(weight)
			self.weights = torch.Tensor(weights)

		if weighting_species:
			gpos, spos = taxa_info['g']
			wfactor_map = {}
			for gid in range(gpos, spos):
				siblings = self.model.CM[gid][spos:].numpy()
				wfactors = np.ones(spos)
				sweights = siblings / 2 + (1 - siblings)
				wfactors = np.concatenate([wfactors, sweights])
				wfactor_map[gid] = wfactors
			self.wfactor_map = wfactor_map

	def forward(self, input_ids, labels=None):
		output = self.model(input_ids, labels=labels)
		return output

	def __shared_step(self, batch, metrics: Dict[str, Callable] = None):
		labels, input_ids = batch

		weights = self.weights
		if self.hparams.weighting_species:
			weights = []
			for label in labels:
				tmp_label = label.cpu()
				gid, sid = np.where(tmp_label == 1)[0][-2:]
				wfactors = self.wfactor_map[gid].copy()
				wfactors[sid] = 1.0
				weights.append(wfactors * self.weights.numpy())
			weights = torch.Tensor(np.array(weights))
		weights = weights.to(input_ids.device)

		logits = self(input_ids, labels)
		loss_fct = nn.BCELoss(weight=weights, reduction='sum')
		loss = loss_fct(logits, labels)

		if metrics is not None:
			scores = {}
			for metric, metric_fct in metrics.items():
				value = metric_fct(logits, labels)
				scores[metric] = value

			return loss, scores

		return loss

	def training_step(self, batch, batch_idx):
		loss = self.__shared_step(batch)
		self.log('train_loss', loss, batch_size=batch[0].shape[0],
				 on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		metrics = {'val_acc': accuracy}
		loss, scores = self.__shared_step(batch, metrics)
		scores.update({'val_loss': loss})
		self.log_dict(scores, batch_size=batch[0].shape[0],
				 on_epoch=True, prog_bar=True, logger=True)
		return loss

	def configure_optimizers(self):
		optimizer = Adam(self.parameters(), 
			lr=self.hparams.learning_rate,
			weight_decay=0.01, 
			eps=self.hparams.adam_epsilon,
			betas=(self.hparams.adam_beta1, self.hparams.adam_beta2)
		)

		if self.hparams.lr_schedule:
			scheduler = InverseSqrtScheduler(optimizer, self.hparams.warmup_steps)
			sch_config = {
				"scheduler": scheduler,
				"interval": "step",
			}
			return {
				"optimizer": optimizer,
				"lr_scheduler": sch_config,
			}
			
		return optimizer
		
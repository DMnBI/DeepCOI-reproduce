import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from transformers import EsmConfig


class OneHotEncoding(nn.Module):

	def __init__(self,
		vocab_size: int):
		super().__init__()
		self.embed_size = vocab_size

	def forward(self, x):
		return F.one_hot(x.to(torch.int64), num_classes=self.embed_size)


class SimpleMLP(nn.Module):

	def __init__(self,
		in_dim: int,
		hid_dim: int,
		out_dim: int,
		dropout: float = 0.1):
		super().__init__()
		self.mlp = nn.Sequential(
			weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
			nn.ReLU(),
			nn.Dropout(dropout),
			weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
		)

	def forward(self, x):
		return self.mlp(x)


class Cnn(nn.Module):

	def __init__(self,
		in_dim: int,
		out_dim: int,
		cnn_width: int = 5,
		dropout: float = 0.1):
		super().__init__()
		self.cnn = weight_norm(nn.Conv1d(
				in_dim, 
				in_dim, 
				cnn_width, padding=cnn_width//2), 
			dim=None
		)
		self.dense = weight_norm(nn.Linear(
				in_dim,
				out_dim),
			dim=None
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		tmp = x.permute(0, 2, 1)
		output = self.cnn(tmp).permute(0, 2, 1)
		output = self.dropout(output)

		logits = self.dense(output)
		return logits


class Gru(nn.Module):

	def __init__(self,
		in_dim: int,
		out_dim: int,
		dropout: float = 0.1):
		super().__init__()
		self.gru = nn.GRU(
			in_dim, 
			in_dim // 2,
			num_layers=2,
			bidirectional=True,
			batch_first=True,
		)
		self.dense = weight_norm(nn.Linear(
				in_dim,
				out_dim),
			dim=None
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		batch_size = x.shape[0]
		hid_size = x.shape[-1] // 2
		zeros = torch.zeros(4, batch_size, hid_size,
			dtype=x.dtype, device=x.device)

		output, _ = self.gru(x, zeros)
		output = self.dropout(output)

		logits = self.dense(output)
		return logits


class CnnGru(nn.Module):

	def __init__(self,
		in_dim: int,
		out_dim: int,
		cnn_width: int = 5,
		dropout: float = 0.1):
		super().__init__()
		self.cnn = weight_norm(nn.Conv1d(
				in_dim, 
				in_dim, 
				cnn_width, padding=cnn_width//2), 
			dim=None
		)
		self.gru = nn.GRU(
			in_dim, 
			in_dim // 2,
			num_layers=2,
			bidirectional=True,
			batch_first=True,
		)
		self.dense = weight_norm(nn.Linear(
				in_dim,
				out_dim),
			dim=None
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		tmp = x.permute(0, 2, 1)
		output = self.cnn(tmp).permute(0, 2, 1)

		batch_size = output.shape[0]
		hid_size = output.shape[-1] // 2
		zeros = torch.zeros(4, batch_size, hid_size,
			dtype=output.dtype, device=output.device)

		output, _ = self.gru(output, zeros)
		output = self.dropout(output)

		logits = self.dense(output)
		return logits
		

class AbstractAutoEncoder(nn.Module):
	def __init__(self,
		embed_size: int,
		latent_size: int):
		super().__init__()
		self.embed_size = embed_size
		self.latent_size = latent_size

		self.encoder = None
		self.decoder = None

	def forward(self, x):
		latents = self.encode(x)
		outputs = self.decode(latents)

		return (outputs, latents)

	def encode(self, x):
		raise NotImplementedError

	def decode(self, x):
		raise NotImplementedError


class SimpleAutoEncoder(AbstractAutoEncoder):
	def __init__(self,
		max_seq_length: int,
		**kwargs):
		super().__init__(**kwargs)
		self.max_seq_length = max_seq_length

		self.encoder = nn.Linear(self.embed_size * max_seq_length, self.latent_size)
		self.decoder = nn.Linear(self.latent_size, self.embed_size * max_seq_length)

	def encode(self, x):
		tmp = x.view(x.shape[0], -1)
		return self.encoder(tmp)

	def decode(self, x):
		outputs = self.decoder(x)
		return outputs.view(outputs.shape[0], self.max_seq_length, self.embed_size)


class mLSTMCell(nn.Module):
	def __init__(self,
		embed_size: int,
		latent_size: int):
		super().__init__()
		project_size = latent_size * 4
		self.wmx = weight_norm(
			nn.Linear(embed_size, latent_size, bias=False))
		self.wmh = weight_norm(
			nn.Linear(latent_size, latent_size, bias=False))
		self.wx = weight_norm(
			nn.Linear(embed_size, project_size, bias=False))
		self.wh = weight_norm(
			nn.Linear(latent_size, project_size, bias=True))

	def forward(self, inputs, state):
		h_prev, c_prev = state
		m = self.wmx(inputs) * self.wmh(h_prev)
		z = self.wx(inputs) + self.wh(m)
		i, f, o, u = torch.chunk(z, 4, 1)
		i = torch.sigmoid(i)
		f = torch.sigmoid(f)
		o = torch.sigmoid(o)
		u = torch.tanh(u)
		c = f * c_prev + i * u
		h = o * torch.tanh(c)

		return h, c


class mLSTM(nn.Module):

	def __init__(self, 
		embed_size: int,
		latent_size: int):
		super().__init__()
		self.mlstm_cell = mLSTMCell(embed_size, latent_size)
		self.hidden_size = latent_size

	def forward(self, inputs, state=None, mask=None):
		batch_size = inputs.size(0)
		seqlen = inputs.size(1)

		if mask is None:
			mask = torch.ones(batch_size, seqlen, 1, dtype=inputs.dtype, device=inputs.device)
		elif mask.dim() == 2:
			mask = mask.unsqueeze(2)

		if state is None:
			zeros = torch.zeros(batch_size, self.hidden_size,
				dtype=inputs.dtype, device=inputs.device)
			state = (zeros, zeros)

		steps = []
		for seq in range(seqlen):
			prev = state
			seq_input = inputs[:, seq, :]
			hx, cx = self.mlstm_cell(seq_input, state)
			seqmask = mask[:, seq]
			hx = seqmask * hx + (1 - seqmask) * prev[0]
			cx = seqmask * cx + (1 - seqmask) * prev[1]
			state = (hx, cx)
			steps.append(hx)

		return torch.stack(steps, 1), (hx, cx)


class mLSTMAutoEncoder(AbstractAutoEncoder):
	def __init__(self,
		**kwargs):
		super().__init__(**kwargs)
		self.encoder = mLSTM(self.embed_size, self.latent_size)
		self.decoder = mLSTMCell(self.embed_size, self.latent_size)

		self.batch_size = None
		self.seqlen = None

	def __set_batch_info(self, x):
		self.batch_size = x.shape[0]
		self.seqlen = x.shape[1]

	def __del_batch_info(self, x):
		self.batch_size = None
		self.seqlen = None

	def encode(self, x):
		self.__set_batch_info(x)
		_, (latents, _) = self.encoder(x)

		return latents

	def decode(self, x):
		hidden_states = ()
		hx = x
		zeros = torch.zeros(self.batch_size, self.latent_size,
			dtype=x.dtype, device=x.device)
		state = (zeros, zeros)

		for pos in range(self.seqlen-1, -1, -1):
			prev = state
			hx, cx = self.decoder(hx, state)
			state = (hx, cx)
			hidden_states = (hx, ) + hidden_states

		self.__del_batch_info(x)

		return torch.stack(hidden_states, dim=1)


class GruAutoEncoder(AbstractAutoEncoder):
	def __init__(self,
		**kwargs):
		super().__init__(**kwargs)
		self.encoder = nn.GRU(self.embed_size, self.latent_size, batch_first=True)
		self.decoder = nn.GRUCell(self.embed_size, self.latent_size)

		self.batch_size = None
		self.seqlen = None

	def __set_batch_info(self, x):
		self.batch_size = x.shape[0]
		self.seqlen = x.shape[1]

	def __del_batch_info(self, x):
		self.batch_size = None
		self.seqlen = None

	def encode(self, x):
		self.__set_batch_info(x)
		_, latents = self.encoder(x)

		return latents.squeeze(0)

	def decode(self, x):
		hidden_states = ()
		hx = x
		zeros = torch.zeros(self.batch_size, self.latent_size,
			dtype=x.dtype, device=x.device)
		state = zeros

		for pos in range(self.seqlen-1, -1, -1):
			prev = state
			hx = self.decoder(hx, state)
			state = hx
			hidden_states = (hx, ) + hidden_states

		self.__del_batch_info(x)

		return torch.stack(hidden_states, dim=1)


class ResNetAutoEncoder(AbstractAutoEncoder):
	def __init__(self,
		**kwargs):
		super().__init__(**kwargs)

	def encode(self, x):
		raise NotImplementedError

	def decode(self, x):
		raise NotImplementedError


class TransformerVAE(nn.Module):
	def __init__(self,
		config: EsmConfig):
		super().__init__()
		self.mu_linear = nn.Linear(config.hidden_size, config.hidden_size)
		self.logstd_linear = nn.Linear(config.hidden_size, config.hidden_size)

		self.decoder_layer = nn.TransformerDecoderLayer(
			d_model=config.hidden_size,
			nhead=config.num_attention_heads,
			dim_feedforward=config.intermediate_size,
			dropout=config.hidden_dropout_prob,
			activation=config.hidden_act,
			layer_norm_eps=config.layer_norm_eps,
			batch_first=True,
		)
		self.decoder = nn.TransformerDecoder(
			decoder_layer=self.decoder_layer,
			num_layers=config.num_hidden_layers,
		)

	def forward(self, x):
		latents, mu, logstd = self.encode(x)
		outputs = self.decode(latents)

		return (outputs, latents, mu, logstd)

	def encode(self, x):
		mu = self.mu_linear(x)
		logstd = self.logstd_linear(x)
		latents = self.reparameterize(mu, logstd)

		return (latents, mu, logstd)

	def decode(self, x):
		outputs = self.decoder(x, x)
		return outputs

	def reparameterize(self, mu, logstd):
		return mu + torch.randn_like(logstd) * torch.exp(logstd)


class VAE(nn.Module):
	def __init__(self,
		AutoEncoder: nn.Module,
		latent_size: int):
		super().__init__()
		self.AE = AutoEncoder
		self.mu_linear = nn.Linear(latent_size, latent_size)
		self.logstd_linear = nn.Linear(latent_size, latent_size)

	def forward(self, x):
		latents, mu, logstd = self.encode(x)
		outputs = self.decode(latents)

		return (outputs, latents, mu, logstd)

	def encode(self, x):
		latents = self.AE.encode(x)
		mu = self.mu_linear(latents)
		logstd = self.logstd_linear(latents)
		latents = self.reparameterize(mu, logstd)

		return (latents, mu, logstd)

	def decode(self, x):
		outputs = self.AE.decode(x)
		return outputs

	def reparameterize(self, mu, logstd):
		return mu + torch.randn_like(logstd) * torch.exp(logstd)

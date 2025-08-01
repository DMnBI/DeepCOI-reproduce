import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import (
    EsmModel,
    EsmForMaskedLM,
    EsmConfig,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.optimization import AdamW

from .lr_scheduler import InverseSqrtScheduler
from .modeling_utils import (
    AbstractAutoEncoder,
    SimpleAutoEncoder,
    mLSTMAutoEncoder,
    GruAutoEncoder,
    TransformerVAE,
    VAE,
)


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class EsmLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x


class EsmAbstractModel(pl.LightningModule):
    def __init__(self,
        model_name_or_path: str, 
        learning_rate: float, 
        warmup_steps: int,
        adam_beta1: float, 
        adam_beta2: float, 
        adam_epsilon: float,
        **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.config = EsmConfig.from_pretrained(
            model_name_or_path, return_dict=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        scheduler = InverseSqrtScheduler(optimizer, self.hparams.warmup_steps)
        sch_config = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": sch_config,
        }


class EsmMLMModel(EsmAbstractModel):
    def __init__(self,
        **kwargs):
        super().__init__(**kwargs)

        self.model = EsmForMaskedLM(self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        ).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, sync_dist=True) 


class AutoEncoderMLMModel(EsmAbstractModel):
    def __init__(self,
        ae_class: AbstractAutoEncoder,
        max_seq_length: int = None,
        bottleneck_size: int = None,
        **kwargs):
        super().__init__(**kwargs)

        self.esm = EsmModel(self.config, add_pooling_layer=False)
        self.ae = None
        if ae_class.__name__ == "SimpleAutoEncoder":
            self.ae = ae_class(max_seq_length, 
                embed_size=self.config.hidden_size, 
                latent_size=bottleneck_size
            )
        else:
            self.ae = ae_class(
                embed_size=self.config.hidden_size, 
                latent_size=bottleneck_size
            )
        self.lm_head = EsmLMHead(self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        esm_outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output, latents = self.ae(esm_outputs[0])
        prediction_scores = self.lm_head(sequence_output)
        esm_outputs.last_hidden_state = sequence_output

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores, latents, esm_outputs[2:])
        output = ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return output

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('train_loss', outputs[0], on_step=True, sync_dist=True)
        return outputs[0]

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('valid_loss', outputs[0], on_step=True, sync_dist=True)


class VaeMLMModel(EsmAbstractModel):
    def __init__(self,
        ae_class: AbstractAutoEncoder = None,
        max_seq_length: int = None,
        bottleneck_size: int = None,
        is_tvae: bool = False,
        **kwargs):
        super().__init__(**kwargs)

        self.esm = EsmModel(self.config, add_pooling_layer=False)
        self.vae = None
        if is_tvae:
            self.vae = TransformerVAE(self.config)
        else:
            self.ae = None
            if ae_class.__name__ == "SimpleAutoEncoder":
                self.ae = ae_class(max_seq_length, 
                    embed_size=self.config.hidden_size, 
                    latent_size=bottleneck_size
                )
            else:
                self.ae = ae_class(
                    embed_size=self.config.hidden_size, 
                    latent_size=bottleneck_size
                )
            self.vae = VAE(self.ae, bottleneck_size)
        self.lm_head = EsmLMHead(self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        esm_outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output, latents, mu, logstd = self.vae(esm_outputs[0])
        prediction_scores = self.lm_head(sequence_output)
        esm_outputs.last_hidden_state = sequence_output

        masked_lm_loss, kl_loss = None, None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            kl_loss = self.KLLoss(mu, logstd)
            batch_size = input_ids.shape[0]
            kl_loss = kl_loss / batch_size

        output = (prediction_scores, latents, esm_outputs[2:])
        output = ((masked_lm_loss, kl_loss) + output) if masked_lm_loss is not None else output

        return output

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('train_loss', outputs[0], on_step=True, sync_dist=True)
        self.log('train_kl_loss', outputs[1], on_step=True, sync_dist=True)
        return outputs[0] + outputs[1]

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('valid_loss', outputs[0], on_step=True, sync_dist=True)
        self.log('valid_kl_loss', outputs[1], on_step=True, sync_dist=True)

    def KLLoss(self, mu, logstd):
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

from typing import Optional, List, Union
import warnings

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.utils import freeze_up_to
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks.scenarios import CLExperience, CLStream

import sys
from small_100.modeling_m2m_100 import M2M100Model
from small_100.tokenization_small100 import SMALL100Tokenizer
from model.small100_config import Small100Config
from core.load_flores200 import create_collate_fn

import model.utils as utils

class LatentReplayTransformer(SupervisedTemplate):
    """Latent Replay.

    This implementations allows for the use of Latent Replay to protect the
    lower level of the model from forgetting.
    """

    def __init__(
        self,
        *,
        model=None,
        criterion=None,
        lr: float = 0.001,
        momentum=0.9,
        weight_decay=0.0005,
        train_epochs: int = 4,
        rm_sz: int = 1500,
        freeze_below_layer: str = "end_features.0",
        latent_layer_num: int = 19,
        subsample_replays: bool = False,
        train_mb_size: int = 16,
        eval_mb_size: int = 16,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        pretrained: bool = False,
        max_seq_len: int = 128,
        warmup_steps: int = 4000,
    ):
        """
        Creates an instance of the LatentReplay strategy.

        :param criterion: The loss criterion to use. Defaults to None, in which
            case the cross entropy loss is used.
        :param lr: The learning rate (SGD optimizer).
        :param momentum: The momentum (SGD optimizer).
        :param weight_decay: The L2 penalty used for weight decay.
        :param train_epochs: The number of training epochs. Defaults to 4.
        :param rm_sz: The size of the replay buffer. The replay buffer is shared
            across classes. Defaults to 1500.
        :param freeze_below_layer: A string describing the name of the layer
            to use while freezing the lower (nearest to the input) part of the
            model. The given layer is not frozen (exclusive). Please ensure this
            layer has a grad function. Defaults to "end_features.0".
        :param latent_layer_num: The number of the layer to use as the Latent
            Replay Layer. Usually this is the same of `freeze_below_layer`.
        :param train_mb_size: The train minibatch size. Defaults to 128.
        :param eval_mb_size: The eval minibatch size. Defaults to 128.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

        warnings.warn(
            "LatentReplay and GenerativeLatentReplay will only recognise "
            "modules defined in __init__. "
            "Modules defined in forward will be ignored."
        )

        if plugins is None:
            plugins = []

        # Model setup
        if not pretrained:
            self.config = Small100Config()
            self.model = M2M100Model(config = self.config, latent_layer_num=latent_layer_num)
        else:
            print("Loading pretrained model...")
            self.config = Small100Config.from_pretrained("alirezamsh/small100")
            self.model = M2M100Model.from_pretrained("alirezamsh/small100", latent_layer_num=latent_layer_num)

        print("Learning rate used: ", lr)
        print("Memory buffer size: ", rm_sz)
        optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        if criterion is None:
            criterion = CrossEntropyLoss()

        self.latent_layer_num = latent_layer_num
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.rm = None
        self.rm_sz = rm_sz
        self.freeze_below_layer = freeze_below_layer
        self.cur_acts: Optional[Tensor] = None
        self.cur_y: Optional[Tensor] = None
        self.cur_attention_mask: Optional[Tensor] = None
        self.cur_decoder_attention_mask: Optional[Tensor] = None
        self.replay_mb_size = 0
        self.subsample_replays = subsample_replays
        self.pretrained = pretrained
        self.max_seq_len = max_seq_len
        self.scheduler = LambdaLR(optimizer, lr_lambda=utils.inverse_sqrt_schedule(warmup_steps))

        super().__init__(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        self._mb_x = None
        self._mb_y = None

    def _before_training_exp(self, **kwargs):
        # Freeze model backbone during subsequent experiences
        if self.pretrained or self.clock.train_exp_counter > 0:
            frozen_layers, frozen_parameters = freeze_up_to(
                self.model, self.freeze_below_layer
            )
            print(f"Frozen layers:\n {frozen_layers}")

            # Adapt the model and optimizer
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-6,
            )

        # super()... will run S.I. and CWR* plugin callbacks
        super()._before_training_exp(**kwargs)

    def make_train_dataloader(
        self, num_workers=0, shuffle=True, pin_memory=True, **kwargs
    ):
        """
        Called after the dataset instantiation. Initialize the data loader.
        """

        print(f"DEBUG: make_train_dataloader called")

        current_batch_mb_size = self.train_mb_size

        if True:  # self.clock.train_exp_counter > 0:
            train_patterns = len(self.adapted_dataset)

            if self.subsample_replays:
                current_batch_mb_size //= 2
            else:
                current_batch_mb_size = train_patterns // (
                    (train_patterns + self.rm_sz) // self.train_mb_size
                )
        print(f"DEBUG: self.rm_size = {self.rm_sz}")
        print(f"DEBUG: current_batch_mb_size = {current_batch_mb_size}")
        print(f"DEBUG: self.replay_mb_size = {self.replay_mb_size}")
        current_batch_mb_size = max(1, current_batch_mb_size)
        self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)

        # Get language info from the dataset
        original_dataset = self.adapted_dataset
        while hasattr(original_dataset, 'dataset'):
            original_dataset = original_dataset.dataset
        # Get language info or default to "eng_Latn" and "fra_Latn"
        src_lang = getattr(original_dataset, 'src_lang', "eng_Latn")
        tgt_lang = getattr(original_dataset, 'tgt_lang', "fra_Latn")

        train_collate_fn = create_collate_fn(src_lang, tgt_lang, self.max_seq_len)
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=current_batch_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn = train_collate_fn,
        )

    # JA: See here for implementing custom loss function:
    # https://github.com/ContinualAI/avalanche/pull/604

    def training_epoch(self, **kwargs):
        print(f"DEBUG: vocab size= {self.config.vocab_size}")
        for mb_it, self.mbatch in enumerate(self.dataloader):
            print(f"DEBUG: mb_it = {mb_it}")
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            self.optimizer.zero_grad()

            # Grab y labels for the current minibatch
            cur_y = self.mb_y.detach().clone().cpu()

            if self.clock.train_exp_counter > 0 and self.rm_sz > 0:
                start = (self.replay_mb_size * mb_it) % self.rm[0].size(0)
                end = (self.replay_mb_size * (mb_it + 1)) % self.rm[0].size(0)

                lat_mb_x = self.rm[0][start:end].to(self.device)
                lat_mb_y = self.rm[1][start:end].to(self.device)
                lat_attention_mask = self.rm[2][start:end].to(self.device)
                lat_decoder_attention_mask = self.rm[3][start:end].to(self.device)

                # Set current y_labels to current minibatch plus replayed examples
                self._mb_y = torch.cat((self.mb_y, lat_mb_y), 0)
                self._decoder_attention_mask = torch.cat(
                    (self.decoder_attention_mask, lat_decoder_attention_mask), dim=0
                )
            else:
                lat_mb_x = None

            # Forward pass. Here we are injecting latent patterns lat_mb_x.
            # lat_mb_x will be None for the very first batch (batch 0), which
            # means that lat_acts.shape[0] == self.mb_x[0].
            self._before_forward(**kwargs)

            # Get mb_x and mb_y embeddings
            mb_x_embeds = self.model.shared(self.mb_x)
            mb_y_embeds = self.model.shared(self.mb_y)
            print(f"DEBUG: mb_x_embeds.shape= {mb_x_embeds.shape}")
            print(f"DEBUG: mb_y_embeds.shape= {mb_y_embeds.shape}")
            # JA:
            if mb_it == 0 and False:
                utils.render_model(
                    lat_mb_x, self.model, self.mb_x, self.clock.train_exp_counter
                )
            if self.clock.train_exp_counter > 0 and self.rm_sz > 0:
                print(f"DEBUG: lat_mb_x.shape= {lat_mb_x.shape if lat_mb_x is not None else 'None'}")
                print(f"DEBUG: lat_mb_y.shape= {lat_mb_y.shape if lat_mb_y is not None else 'None'}")

            outputs = self.model(
                inputs_embeds=mb_x_embeds,
                attention_mask=self.attention_mask,
                decoder_inputs_embeds=mb_y_embeds,
                decoder_attention_mask=self.decoder_attention_mask,
                latent_encoder_input=lat_mb_x,
                latent_attention_mask = lat_attention_mask if lat_mb_x is not None else None,
                return_lat_acts=True,
                return_dict = True,
            )

            lat_acts = outputs.latent_acts
            self.mb_output = outputs.logits
            # print(f"DEBUG: lat_acts= {lat_acts.shape}")
            print(f"DEBUG: mb_output= {self.mb_output.shape}")

            if self.clock.train_exp_epochs == 0 and self.rm_sz > 0:
                # On the first epoch only: store latent activations. Those
                # activations will be used to update the replay buffer.
                lat_acts = lat_acts.detach().clone().cpu()

                if mb_it == 0:
                    self.cur_acts = lat_acts
                    self.cur_y = cur_y
                    self.cur_attention_mask = self.attention_mask.detach().clone().cpu()
                    self.cur_decoder_attention_mask = self.decoder_attention_mask.detach().clone().cpu()
                else:
                    self.cur_acts = torch.cat((self.cur_acts, lat_acts), 0)
                    self.cur_y = torch.cat((self.cur_y, cur_y), 0)
                    self.cur_attention_mask = torch.cat(
                        (self.cur_attention_mask, self.attention_mask.detach().clone().cpu()), dim=0
                    )
                    self.cur_decoder_attention_mask = torch.cat(
                        (self.cur_decoder_attention_mask, self.decoder_attention_mask.detach().clone().cpu()), dim=0
                    )

            self._after_forward(**kwargs)
            # Loss & Backward
            # We don't need to handle latent replay, as self.mb_y already
            # contains both current and replay labels.
            self._mb_y = self.mb_y.to(self.mb_output.device)
            print(f"DEBUG: self.mb_y.shape= {self.mb_y.shape}")
            self.loss = self._criterion(self.mb_output.reshape(-1, self.config.vocab_size), self.mb_y.reshape(-1))
            with open("loss.txt", "a") as f:
                f.write(f"{mb_it},{self.loss.item()}\n")
            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

            print(f"[{mb_it}] MPS allocated: {torch.mps.current_allocated_memory() / 1e6:.2f} MB")

    def _after_training_exp(self, **kwargs):
        if self.rm_sz == 0:
            super()._after_training_exp(**kwargs)
            return

        h = min(
            self.rm_sz // (self.clock.train_exp_counter + 1),
            self.cur_acts.size(0),
        )

        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]
        print("DEBUG: Adding patterns to replay memory")
        print(f"DEBUG: cur_acts shape = {self.cur_acts.shape}")
        print(f"DEBUG: cur_y shape = {self.cur_y.shape}")
        print(f"DEBUG: attention_mask shape = {self.cur_attention_mask.shape}")
        print(f"DEBUG: decoder_attention_mask shape = {self.cur_decoder_attention_mask.shape}")
        rm_add = [self.cur_acts[idxs_cur], self.cur_y[idxs_cur], self.cur_attention_mask[idxs_cur], self.cur_decoder_attention_mask[idxs_cur]]

        # replace patterns in random memory
        if self.clock.train_exp_counter == 0:
            self.rm = rm_add
        else:
            existing_size = self.rm[0].size(0)
            if existing_size < self.rm_sz:
                remaining_size = self.rm_sz - existing_size
                to_append = min(h, remaining_size)
                self.rm[0] = torch.cat((self.rm[0], rm_add[0][:to_append]), dim=0)
                self.rm[1] = torch.cat((self.rm[1], rm_add[1][:to_append]), dim=0)
                self.rm[2] = torch.cat((self.rm[2], rm_add[2][:to_append]), dim=0)
                self.rm[3] = torch.cat((self.rm[3], rm_add[3][:to_append]), dim=0)

            else:
                # Buffer is full, replace h random indices
                idxs_to_replace = torch.randperm(self.rm[0].size(0))[:h]
                self.rm[0][idxs_to_replace] = rm_add[0]
                self.rm[1][idxs_to_replace] = rm_add[1]
                self.rm[2][idxs_to_replace] = rm_add[2]
                self.rm[3][idxs_to_replace] = rm_add[3]

        self.cur_acts = None
        self.cur_y = None
        self.cur_attention_mask = None
        self.cur_decoder_attention_mask = None
        # Runs plugin callbacks
        super()._after_training_exp(**kwargs)

    @property
    def mb_x(self):
        """Getter for encoder input (source language tokens)."""
        return self._mb_x

    @property
    def mb_y(self):
        """Getter for decoder input (target language tokens)."""
        return self._mb_y

    @property
    def attention_mask(self):
        """Getter for attention mask."""
        return self._attention_mask

    @property
    def decoder_attention_mask(self):
        """Getter for decoder attention mask."""
        return self._decoder_attention_mask

    def _unpack_minibatch(self):
        """Unpacks minibatches for an encoder-decoder Transformer in machine translation."""
        self._mb_x, self._mb_y, self._attention_mask, self._decoder_attention_mask = self._process_batch(self.mbatch)

        # Debugging to confirm correct unpacking
        print(f"DEBUG: mb_x shape={self._mb_x.shape}, mb_y shape={self._mb_y.shape}")
        print(f"DEBUG: attention mask shape={self._attention_mask.shape}, decoder attention mask shape={self._decoder_attention_mask.shape}")

    def _process_batch(self, batch):
        """Helper function to process a batch into tensors with attention masks."""
        batch = batch.to(self.device) if isinstance(batch, torch.Tensor) else batch

        if isinstance(batch, dict):
            # Move all batch elements to device
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            return (
                batch["input_ids"][:, 0, :], batch["input_ids"][:, 1, :],
                batch["attention_mask"][:, 0, :], batch["attention_mask"][:, 1, :]
            )

        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return (batch[0].to(self.device), batch[1].to(self.device), None, None)

        elif isinstance(batch, torch.Tensor) and batch.dim() == 3 and batch.shape[1] == 2:
            return (batch[:, 0, :].to(self.device), batch[:, 1, :].to(self.device), None, None)

        else:
            raise ValueError(f"Unexpected minibatch format: {type(batch)}, shape={batch.shape if isinstance(batch, torch.Tensor) else 'N/A'}")

    @torch.no_grad()
    def eval(self, exp_list: Union[CLExperience, CLStream], **kwargs):
        """
        Evaluates the model using BLEU score over the test set.
        """
        self.model.eval()  # Set model to evaluation mode

        tokenizer = SMALL100Tokenizer()  # Ensure same tokenizer used during training
        results = {}
        for experience in exp_list:
            bleu_metric = utils.BLEUMetric()  # Initialize BLEU metric
            src_lang = experience.dataset.src_lang
            tgt_lang = experience.dataset.tgt_lang
            eval_collate_fn = create_collate_fn(src_lang, tgt_lang, self.max_seq_len)
            print(f"DEBUG: src={src_lang}, tgt={tgt_lang}")

            eval_dataloader = DataLoader(
                experience.dataset,
                batch_size=self.eval_mb_size,
                shuffle=False,
                collate_fn=eval_collate_fn,
            )
            for mb_it, batch in enumerate(eval_dataloader):
                # print(f"DEBUG: mb_it = {mb_it}")

                mb_x = batch["input_ids"][:, 0, :].to(self.device)  # Encoder input
                mb_y = batch["input_ids"][:, 1, :].to(self.device)  # Target sequence
                attention_mask = batch["attention_mask"][:, 0, :].to(self.device)
                decoder_attention_mask = batch["attention_mask"][:, 1, :].to(self.device)

                self.model.to(self.device)

                outputs = self.model(
                    input_ids=mb_x,
                    attention_mask=attention_mask,
                    decoder_input_ids=mb_y,  # Decoder gets previous tokens as input
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=True,
                )

                # print(f"DEBUG: mb_y.shape= {mb_y.shape}")

                logits = outputs.logits
                predicted_token_ids = logits.argmax(dim=-1) # only used during eval
                print(f"DEBUG: predicted_token_ids.shape= {predicted_token_ids.shape}")

                predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predicted_token_ids]
                references = [tokenizer.decode(ref, skip_special_tokens=True) for ref in mb_y]

                # 🔍 DEBUG PRINT to verify output vs target
                print(f"\n--- DEBUG BATCH {mb_it} ---")
                for i in range(min(2, len(predictions))):  # Print up to 2 examples
                    print(f"🟦 Source:{tokenizer.decode(mb_x[i], skip_special_tokens=True)}")
                    print(f"🟩 Predicted:{predictions[i]}")
                    print(f"🟥 Target:{references[i]}")
                    print()

                # append to predictions.txt
                # with open("predictions.txt", "a") as f:
                #     for pred in predictions:
                #         f.write(f"{pred}\n")

                for ref, pred in zip(references, predictions):
                    bleu_metric.update(ref, pred)

            avg_bleu = bleu_metric.compute()
            results[src_lang] = avg_bleu

        return results
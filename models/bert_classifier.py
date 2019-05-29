from typing import *

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides


import torch
from allennlp.modules import Seq2VecEncoder
from overrides import overrides

class BertSentencePooler(Seq2VecEncoder):
    def __init__(self, vocab, dim):
        super().__init__(vocab)
        self.bert_dim = dim

    def forward(self, embs: torch.tensor,
                mask: torch.tensor = None) -> torch.tensor:
        # extract first token tensor
        #return torch.transpose(embs, 1, 2)
        return embs[:, 0]

    @overrides
    def get_output_dim(self) -> int:
        return self.bert_dim

class BertClassifier(Model):
    def __init__(self, args, out_sz: int,
                 vocab: Vocabulary):
        super().__init__(vocab)

        # init word embedding
        bert_embedder = PretrainedBertEmbedder(
            # requires_grad=True,
            pretrained_model="bert-base-uncased",
            top_layer_only=True,  # conserve memory
        )
        self.word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                      # we'll be ignoring masks so we'll need to set this to True
                                                      allow_unmatched_keys=True)
        aa = self.word_embeddings.get_output_dim()
        self.encoder = BertSentencePooler(vocab, self.word_embeddings.get_output_dim())
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        # state = self.encoder(embeddings, mask)[:, :, -1].squeeze(-1)
        class_logits = self.projection(state)

        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)
        self.accuracy(class_logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
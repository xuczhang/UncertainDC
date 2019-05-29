# TODO:
from typing import *

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


class TransformerClassifier(Model):
    def __init__(self, args, out_sz: int,
                 vocab: Vocabulary):
        super().__init__(vocab)

        # prepare embeddings
        token_embedding = Embedding(num_embeddings=args.max_vocab_size + 2,
                                    embedding_dim=300, padding_index=0)
        self.word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        # self.encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(self.word_embeddings.get_output_dim(),
        #                                                              hidden_size=64, bidirectional=True, batch_first=True))
        self.encoder = StackedSelfAttentionEncoder(input_dim=self.word_embeddings.get_output_dim(),
                                                   hidden_dim=128,
                                                   projection_dim=128,
                                                   feedforward_hidden_dim=128,
                                                   num_layers=1,
                                                   num_attention_heads=8)
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)[:, -1, :].squeeze(1)
        class_logits = self.projection(state)

        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)

        # accuracy
        # pred = (torch.max(class_logits, 1)[1].view(label.size()).data).tolist()
        # from eval import show_results
        # # f1 = show_results(label.cpu().numpy().tolist(), pred)[1]
        self.accuracy(class_logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

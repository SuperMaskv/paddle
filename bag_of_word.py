import paddle.nn as nn
import paddle
import paddlenlp


class BoWModel(nn.Layer):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        emb_dim = self.embedder.embedding_dim
        self.encoder = paddlenlp.seq2vec.BoWEncoder(emb_dim=emb_dim)
        self.cos_sim_fn = nn.CosineSimilarity(axis=-1)

    def forward(self, text):
        embedded_text = self.embedder(text)
        return self.encoder(embedded_text)

    def get_cos_sim(self, text_a, text_b):
        return self.cos_sim_fn(self.forward(text_a), self.forward(text_b))

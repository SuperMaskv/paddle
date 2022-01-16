import paddle
from paddlenlp.embeddings import TokenEmbedding
from data import Tokenizer
from bag_of_word import BoWModel


def main():
    token_embedding = TokenEmbedding()
    tokenizer = Tokenizer()
    tokenizer.set_vocab(token_embedding.vocab)
    model = BoWModel(token_embedding)
    text_a_ids = paddle.to_tensor([tokenizer.text_to_ids("我是你爹")])
    text_b_ids = paddle.to_tensor([tokenizer.text_to_ids("我是你爸爸")])
    print(model.get_cos_sim(text_a_ids, text_b_ids))


if __name__ == '__main__':
    main()

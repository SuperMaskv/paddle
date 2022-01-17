import paddle
from paddlenlp.embeddings import TokenEmbedding
from data import Tokenizer
from bag_of_word import BoWModel


def main():
    token_embedding = TokenEmbedding()
    tokenizer = Tokenizer()
    tokenizer.set_vocab(token_embedding.vocab)
    model = BoWModel(token_embedding)
    text_a = "我是你爷爷的孙子的爸爸"
    text_b = "我是你儿子的爷爷"
    text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])
    text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])
    print("{}\t与\t{}\t的相似度为{}".format(text_a, text_b, model.get_cos_sim(text_a_ids, text_b_ids)))


if __name__ == '__main__':
    main()

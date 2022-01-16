import paddle
from paddlenlp.embeddings import TokenEmbedding
from data import Tokenizer
from bag_of_word import BoWModel


def main():
    token_embedding = TokenEmbedding()
    tokenizer = Tokenizer()
    tokenizer.set_vocab(token_embedding.vocab)
    model = BoWModel(token_embedding)
    print(model.get_cos_sim(tokenizer.text_to_ids("我是你爹"), tokenizer.text_to_ids("我是你爸爸")))


if __name__ == '__main__':
    main()

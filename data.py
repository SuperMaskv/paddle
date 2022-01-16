import collections

import numpy as np
import jieba
import paddle

from collections import defaultdict
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab


class Tokenizer(object):
    def __init__(self):
        self.vocab = None
        self.vocab_dict = {}
        self.vocab_path = 'vocab.txt'
        self.tokenizer = jieba
        self.PAD_TOKEN = '[PAD]'
        self.UNK_TOKEN = '[UNK]'

    def set_vocab(self, vocab):
        self.vocab_dict = vocab
        self.tokenizer = JiebaTokenizer(vocab=vocab)

    def build_vocab(self, sentences):
        word_count = defaultdict(lambda: 0)
        for text in sentences:
            words = jieba.lcut(text)
            for word in words:
                word = word.strip()
                if word.strip() != '':
                    word_count[word] += 1

        word_id = 0
        for word, num in word_count.items():
            if num < 5:
                continue
            self.vocab_dict[word] = word_id
            word_id += 1
        self.vocab_dict[self.PAD_TOKEN] = word_id
        self.vocab_dict[self.UNK_TOKEN] = word_id + 1
        self.vocab = Vocab.from_dict(self.vocab_dict, pad_token=self.PAD_TOKEN, unk_token=self.UNK_TOKEN)
        self.dump_vocab()
        self.tokenizer = JiebaTokenizer(vocab=self.vocab)
        return self.vocab_dict

    def dump_vocab(self):
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            for word in self.vocab.token_to_idx:
                f.write(word + "\n")

    def text_to_ids(self, text):
        input_ids = []
        unk_token_id = self.vocab_dict[self.UNK_TOKEN]
        for token in self.tokenizer.cut(text):
            token_id = self.vocab.token_to_indx.get(token, unk_token_id)
            input_ids.append(token_id)
        return input_ids

    def convert_example(self, example, is_test=False):
        input_ids = self.text_to_ids(example['text'])

        if not is_test:
            label = np.array(example['label'], dtype='int64')
            return input_ids, label
        else:
            return input_ids


def create_dataloader(
        dataset,
        trans_fn=None,
        mode='train',
        batch_size=1,
        pad_token_id=0
):
    if trans_fn:
        dataset = dataset.map(trans_fn, lazy=True)
    shuffle = True if mode == 'train' else False
    sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),
        Stack(dtype='int64')
    ): [data for data in fn(samples)]

    dataloader = paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        return_list=True,
        collate_fn=batchify_fn
    )
    return dataloader

import time

import paddlenlp.transformers
from paddlenlp.datasets import load_dataset
import numpy as np
from functools import partial
from paddlenlp.data import Stack, Pad, Tuple


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    query, title = example['query'], example['title'],
    encoded_inputs = tokenizer(text=query, text_pair=title, max_seq_len=max_seq_length)
    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']

    if not is_test:
        label = np.array([example['label']], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def main():
    train_ds, dev_ds = load_dataset("lcqmc", splits=['train', 'dev'])
    tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained("ernie-gram-zh")
    input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)
    trans_fn = partial(convert_example, tokenizer=tokenizer, max_seq_len=512)




if __name__ == '__main__':
    main()

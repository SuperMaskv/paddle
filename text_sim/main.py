import time

import paddle.io
import paddlenlp.transformers
from paddlenlp.datasets import load_dataset
import numpy as np
from functools import partial
from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.transformers import LinearDecayWithWarmup
from point_wise import PointwiseMatching
import paddle.nn as nn


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


@paddle.no_grad()
def evaluate(model: nn.Layer, criterion, metric: paddle.metric.Metric, data_loader, phase='dev'):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)


def main():
    train_ds, dev_ds = load_dataset("lcqmc", splits=['train', 'dev'])
    tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained("ernie-gram-zh")
    input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)
    trans_fn = partial(convert_example, tokenizer=tokenizer, max_seq_len=512)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(pad_val=tokenizer.pad_token_id),
        Pad(pad_val=tokenizer.pad_token_type_id),
        Stack()
    ): [data for data in fn(samples)]
    # 构建训练集数据加载器
    batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)
    train_dataloader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_fn),
        batch_sampler=batch_sampler,
        shuffle=True,
        collate_fn=batchify_fn
    )
    # 构建验证集数据加载器
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=True)
    dev_dataloader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_fn),
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        shuffle=True
    )

    pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained("ernie-gram-zh")
    model = PointwiseMatching(pretrained_model)

    epochs = 3
    num_training_steps = len(train_dataloader) * epochs

    lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

    decay_params = [p.name for n, p in model.named_parameters() if not any(n in nd for nd in ['bias', 'norm'])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params
    )

    criterion = paddle.nn.loss.CrossEntropyLoss()

    metric = paddle.metric.Accuracy()


if __name__ == '__main__':
    main()

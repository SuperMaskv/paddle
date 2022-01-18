import os
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
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase, np.mean(losses), accu))
    model.train()
    metric.reset()


def main():
    train_ds, dev_ds = load_dataset("lcqmc", splits=['train', 'dev'])
    tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained("ernie-gram-zh")
    trans_fn = partial(convert_example, tokenizer=tokenizer, max_seq_length=512)
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
        return_list=True,
        collate_fn=batchify_fn
    )
    # 构建验证集数据加载器
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=True)
    dev_dataloader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_fn),
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True
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
    global_step = 0
    tic_train = time.time()

    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            input_ids, token_type_ids, labels = batch
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1

            if global_step % 10 == 0:
                print(
                    'global step: %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f speed: %.2f step/s'
                    % (global_step, epoch, step, loss, acc, 10 / (time.time() - tic_train))
                )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % 100 == 0:
                evaluate(model, criterion, metric, dev_dataloader)
    save_dir = os.path.join("checkpoint", "model %d" % global_step)
    os.mkdir(save_dir)

    save_param_dir = os.path.join(save_dir, "model_state.pdparams")
    paddle.save(model.state_dict(), save_param_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    main()

# Transformers-ru

A list of pretrained Transformer models for the Russian language (including multilingual models).

Code for the model using and visualisation is from the following repos:
* [https://github.com/huggingface/pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
* [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)

## Models

There are models form:
* [DeepPavlov project](http://docs.deeppavlov.ai/en/master/features/models/bert.html)
* [Hugging Face repository](https://github.com/huggingface/pytorch-transformers)

| Model description | Config | Vocabulary | Model |
|-|-|-|-|
|BERT-Base, Multilingual Cased: 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters|[[amazon]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json)|[[amazon]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt)|[[amazon]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin)|
|BERT-Base, Multilingual Uncased: 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters|[[amazon]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json)|[[amazon]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt)|[[amazon]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin)|
|RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters|<td colspan="3">[[deeppavlov]](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v1.tar.gz)</td>|
|SlavicBERT, Slavic (bg, cs, pl, ru), cased, 12-layer, 768-hidden, 12-heads, 180M parameters|<td colspan="3">[[deeppavlov]](http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12.tar.gz)</td>|

## Converting TensorFlow models to PyTorch

Downloading and converting the DeepPavlov model:

```bash
wget 'http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v1.tar.gz'
tar -xzf rubert_cased_L-12_H-768_A-12_v1.tar.gz
python3 convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path rubert_cased_L-12_H-768_A-12_v1/bert_model.ckpt \
    --bert_config_file rubert_cased_L-12_H-768_A-12_v1/bert_config.json \
    --pytorch_dump_path rubert_cased_L-12_H-768_A-12_v1/bert_model.bin
```

## Visualization

The attention-head visualization from BertViz:
[[Notebook]](https://github.com/vlarine/transformers-ru/model_view_bert.ipynb)


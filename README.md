# Transformers-ru

A list of pretrained Transformer models for the Russian language (including multilingual models).

Code for the model using and visualisation is from the following repos:
* [https://github.com/huggingface/pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
* [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)

## Models

There are models form:
* [DeepPavlov project](http://docs.deeppavlov.ai/en/master/features/models/bert.html)
* [Hugging Face repository](https://github.com/huggingface/pytorch-transformers)
* [Facebook research](https://github.com/facebookresearch/XLM/)
* [Facebook's fairseq](https://github.com/pytorch/fairseq)
* [Denis Antyukhov](https://towardsdatascience.com/pre-training-bert-from-scratch-with-cloud-tpu-6e2f71028379) [Google Colab code](https://colab.research.google.com/drive/1nVn6AFpQSzXBt8_ywfx6XR8ZfQXlKGAz)
* [Russian RuBERTa](https://github.com/vlarine/ruberta)

| Model description | # params | Config | Vocabulary | Model | BPE codes |
|-|-|:-|:-|:-|:-|
|BERT-Base, Multilingual Cased: 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters|170M|[[huggingface] 1K](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json)|[[huggingface] 973K](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt)|[[huggingface] 682M](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin)|
|BERT-Base, Multilingual Uncased: 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters|160M|[[huggingface] 1K](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json)|[[huggingface] 852K](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt)|[[huggingface] 642M](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin)|
|RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters |170M|||[[deeppavlov] 636M](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v1.tar.gz)|
|SlavicBERT, Slavic (bg, cs, pl, ru), cased, 12-layer, 768-hidden, 12-heads, 180M parameters|170M|||[[deeppavlov] 636M](http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12.tar.gz)|
|XLM (MLM) 15 languages|237M|[[huggingface] 1K](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-config.json)|[[huggingface] 2,9M](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-vocab.json)<br/>[[facebook] 1,5M](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15)|[[huggingface] 1,3G](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-pytorch_model.bin)<br/>[[facebook] 1,3G](https://dl.fbaipublicfiles.com/XLM/mlm_xnli15_1024.pth)|[[huggingface] 1,4M](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-merges.txt)<br/>[[facebook] 1,4M](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15)|
|XLM (MLM+TLM) 15 languages|237M|[[huggingface] 1K](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-config.json)|[[huggingface] 2,9M](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-vocab.json)<br/>[[facebook] 1,5M](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15)|[[huggingface] 661M](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-pytorch_model.bin)<br/>[[facebook] 665M](https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth)|[[huggingface] 1,4M](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-merges.txt)<br/>[[facebook] 1,4M](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15)|
|XLM (MLM) 17 languages|||[[facebook] 2,9M](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_17)|[[facebook] 1,1G](https://dl.fbaipublicfiles.com/XLM/mlm_17_1280.pth)|[[facebook] 2,9M](https://dl.fbaipublicfiles.com/XLM/codes_xnli_17)|
|XLM (MLM) 100 languages|||[[facebook] 3,0M](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_100)|[[facebook] 1,1G](https://dl.fbaipublicfiles.com/XLM/mlm_100_1280.pth)|[[facebook] 2,9M](https://dl.fbaipublicfiles.com/XLM/codes_xnli_100)|
|Denis Antyukhov BERT-Base, Russian, Uncased, 12-layer, 768-hidden, 12-heads|176M|||[[bert_resourses] 1,9G](https://storage.googleapis.com/bert_resourses/russian_uncased_L-12_H-768_A-12.zip)|
|Facebook-FAIR's WMT'19 en-ru||||[[fairseq] 12G](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz)|
|Facebook-FAIR's WMT'19 ru-en||||[[fairseq] 12G](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz)|
|Facebook-FAIR's WMT'19 ru||||[[fairseq] 2,1G](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.gz)|
|Russian RuBERTa||||[[Google Drive] 247M](https://drive.google.com/open?id=1WYpuSCL8oEtW65HIN1izsN_cR5Mizqmd)|

## Converting TensorFlow models to PyTorch

Downloading and converting the DeepPavlov model:

```bash
$ wget 'http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v1.tar.gz'
$ tar -xzf rubert_cased_L-12_H-768_A-12_v1.tar.gz
$ python3 convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path rubert_cased_L-12_H-768_A-12_v1/bert_model.ckpt \
    --bert_config_file rubert_cased_L-12_H-768_A-12_v1/bert_config.json \
    --pytorch_dump_path rubert_cased_L-12_H-768_A-12_v1/bert_model.bin
```

## Models comparison

There are [scripts](scripts) to train and evaluate models on [the Sber SQuAD](http://docs.deeppavlov.ai/en/master/features/models/squad.html) dataset for the russian language [[download dataset]](http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz).

Comparision of BERT models trained on the Sber SQuAD dataset:

| Model | EM (dev) | F-1 (dev) |
|-|-|-|
|BERT-Base, Multilingual Cased|64.85|83.68|
|BERT-Base, Multilingual Uncased|64.73|83.25|
|RuBERT|**66.38**|**84.58**|
|SlavicBERT|65.23|83.68|
|RuBERTa-base|59.45|78.60|

## Visualization

The attention-head view visualization from BertViz:
![Attention-head view](https://github.com/vlarine/transformers-ru/blob/master/img/head.png)

[[Notebook]](https://github.com/vlarine/transformers-ru/blob/master/head_view_bert.ipynb)


The model view visualization from BertViz:
![Model view](https://github.com/vlarine/transformers-ru/blob/master/img/model.jpg)

[[Notebook]](https://github.com/vlarine/transformers-ru/blob/master/model_view_bert.ipynb)

The neuron view visualization from BertViz:
![Neuron view](https://github.com/vlarine/transformers-ru/blob/master/img/neuron.png)

[[Notebook]](https://github.com/vlarine/transformers-ru/blob/master/neuron_view_bert.ipynb)

## Generative models

### GPT-2 models

#### Mikhail Grankin's model

Code: [https://github.com/mgrankin/ru_transformers](https://github.com/mgrankin/ru_transformers)

Download models:
```
pip install awscli
aws s3 sync --no-sign-request s3://models.dobro.ai/gpt2/ru/unfreeze_all gpt2
```

#### Vladimir Larin's model

* Code: [https://github.com/vlarine/ruGPT2](https://github.com/vlarine/ruGPT2)
* Model: [gpt2_345m.tgz (4,2G)](https://drive.google.com/file/d/1hp21DmAoeq6tKoUGLEK8NtPRJVWdz_dH/view?usp=sharing)

## RNN Models

There are some RNN models for russian language.

### ELMo

#### [DeepPavlov](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html)

* ELMo on Russian Wikipedia: [[config]](https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/configs/elmo_embedder/elmo_ru_wiki.json), [[model]](http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz)
* ELMo on Russian WMT News: [[config]](https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/configs/elmo_embedder/elmo_ru_news.json), [[model]](http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz)
* ELMo on Russian Twitter: [[config]](https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/configs/elmo_embedder/elmo_ru_twitter.json), [[model]](http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz)

#### [RusVectōrēs](http://rusvectores.org/en/models/)

* RNC and Wikipedia. December 2018 (tokens): [[model]](http://vectors.nlpl.eu/repository/11/195.zip)
* RNC and Wikipedia. December 2018 (lemmas): [[model]](http://vectors.nlpl.eu/repository/11/196.zip)
* Taiga 2048. December 2019 (lemmas): [[model]](http://vectors.nlpl.eu/repository/20/199.zip)

### ULMFit

* [Pavel Pleskov model](https://github.com/ppleskov/Russian-Language-Model): [[model]](https://drive.google.com/open?id=1gtIfMcu7q44q3aViepWE63WgsdY2Bjvn)
* [mamamot model](https://github.com/mamamot/Russian-ULMFit/): [[model]](https://drive.google.com/open?id=1_d4XCMMWdIZt57JJyH34bzY2gRSB7KTE)

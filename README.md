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

| Model description | Config | Vocabulary | Model | BPE codes |
|-|:-|:-|:-|:-|
|BERT-Base, Multilingual Cased: 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin)|
|BERT-Base, Multilingual Uncased: 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin)|
|RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters |||[[deeppavlov]](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v1.tar.gz)|
|SlavicBERT, Slavic (bg, cs, pl, ru), cased, 12-layer, 768-hidden, 12-heads, 180M parameters|||[[deeppavlov]](http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12.tar.gz)|
|XLM (MLM) 15 languages|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-config.json)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-vocab.json)<br/>[[facebook]](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-pytorch_model.bin)<br/>[[facebook]](https://dl.fbaipublicfiles.com/XLM/mlm_xnli15_1024.pth)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-merges.txt)<br/>[[facebook]](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15)|
|XLM (MLM+TLM) 15 languages|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-config.json)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-vocab.json)<br/>[[facebook]](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-pytorch_model.bin)<br/>[[facebook]](https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth)|[[huggingface]](https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-merges.txt)<br/>[[facebook]](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15)|
|XLM (MLM) 17 languages||[[facebook]](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_17)|[[facebook]](https://dl.fbaipublicfiles.com/XLM/mlm_17_1280.pth)|[[facebook]](https://dl.fbaipublicfiles.com/XLM/codes_xnli_17)|
|XLM (MLM) 100 languages||[[facebook]](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_100)|[[facebook]](https://dl.fbaipublicfiles.com/XLM/mlm_100_1280.pth)|[[facebook]](https://dl.fbaipublicfiles.com/XLM/codes_xnli_100)|
|Denis Antyukhov BERT-Base, Russian, Uncased, 12-layer, 768-hidden, 12-heads|||[[bert_resourses]](https://storage.googleapis.com/bert_resourses/russian_uncased_L-12_H-768_A-12.zip)|
|Facebook-FAIR's WMT'19 en-ru|||[[fairseq]](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz)|
|Facebook-FAIR's WMT'19 ru-en|||[[fairseq]](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz)|
|Facebook-FAIR's WMT'19 ru|||[[fairseq]](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.gz)|

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





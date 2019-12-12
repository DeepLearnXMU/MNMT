# MNMT
## Requirements

```bash
pip install -r requirements.txt
```

### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
python train.py \
    -data data/demo \
    -save_model demo-model \
    -path_to_train_img_feats deme-img \
    -path_to_valid_img_feats deme-val-img \
    -optim adam \
    -learning_rate 0.003 \
    -use_nonlinear_projection \
    -start_checkpoint_at 10 \
    -image_feat_type local \
    -encoder_type brnn \
    -enc_layers 1 \
    -dec_layers 1 \
    -global_attention mlp \
    -rnn_type GRU \
    -dropout 0.3 \
    -bi_attention 1 \
    -co_attention 1 \
    -language en_de

```

You can also add `-gpuid 1` to use GPU 1.

### Step 3: Translate

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.


## Citation


## Note
Most of this code and data are borrowed from:

```
@InProceedings{CalixtoLiu2017EMNLP,
  Title                    = {{Incorporating Global Visual Features into Attention-Based Neural Machine Translation}},
  Author                   = {Iacer Calixto and Qun Liu},
  Booktitle                = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  Year                     = {2017},
  Address                  = {Copenhagen, Denmark},
  Url                      = {http://aclweb.org/anthology/D17-1105}
}
```

```
@InProceedings{CalixtoLiuCampbell2017ACL,
  author    = {Calixto, Iacer  and  Liu, Qun  and  Campbell, Nick},
  title     = {{Doubly-Attentive Decoder for Multi-modal Neural Machine Translation}},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {1913--1924},
  url       = {http://aclweb.org/anthology/P17-1175}
}
```

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

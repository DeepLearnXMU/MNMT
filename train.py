#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import time
import random
import re
from datetime import datetime

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts
import tables
import numpy  as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.train_mm_opts(parser)

opt = parser.parse_args()

    
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    


if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


if torch.initial_seed()>sys.maxsize:
    torch.manual_seed(torch.initial_seed()%sys.maxsize)
print('*'*20,torch.initial_seed(),'*'*20)
if torch.initial_seed()>sys.maxsize:
    print('*'*20,torch.initial_seed(),'*'*20)
    raise AssertionError

torch_seed=torch.initial_seed()
if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        opt.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"),
        comment="Onmt")

progress_step = 0


def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        progress_step(int): the progress step.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", writer, lr, progress_step)
        report_stats = onmt.Statistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        global max_src_in_batch, max_tgt_in_batch

        def batch_size_fn(new, count, sofar):
            global max_src_in_batch, max_tgt_in_batch
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            max_src_in_batch = max(max_src_in_batch,  len(new.src) + 2)
            max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def make_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)

    if use_gpu(opt):
        compute.cuda()

    return compute



def train_model(model, fields, optim, data_type,
                 model_opt,train_img_feats=None,valid_img_feats=None):
                 
    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt,
                                   train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
                           trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count,
                           train_img_feats=train_img_feats,
                           valid_img_feats=valid_img_feats,
                           image_feat_type=opt.image_feat_type,
                           batch_size = opt.batch_size,
                           valid_batch_size = opt.valid_batch_size)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)

    best_BLEU=0
    best_model_name=''
    model_name_list=[]
    cnt=0
    # if os.path.exists(opt.save_model):
    #     os.remove(opt.save_model)
    #     os.mkdir(opt.save_model)
    outfile=open(os.path.join(os.path.dirname(opt.save_model),'BLEU_score.txt'),'w',encoding='utf-8')
    random.seed()
    torch.manual_seed(random.randint(0,sys.maxsize))
    print('start  interate',torch.initial_seed())
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('starting epoch',epoch)

        # 1. Train for one epoch on the training set.
        train_iter = make_dataset_iter(lazily_load_dataset("train"),
                                       fields, opt)
        train_stats = trainer.train(train_iter, epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                       fields, opt,
                                       is_train=False)
        valid_stats = trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)
        if opt.tensorboard:
            train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            model_name=trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)
            model_name_list.append(model_name)
            print(model_name)
            if opt.language is not None:
                language_src=opt.language.split('_')[0]
                language_tgt=opt.language.split('_')[1]
            else:
                language_src='en'
                language_tgt='de'
            # if str(opt.data).__contains__('de_en'):
            #     src_test ='../description/bpe/test_2016_flickr.lc.norm.tok.bpe.'+language_src
            #     src_test2 ='../description/bpe/test_2017_flickr.lc.norm.tok.bpe.'+language_src
            #     src_test3 ='../description/bpe/test_2017_mscoco.lc.norm.tok.bpe.'+language_src
            #     tgt_test='../description/norm_tok_lc/test_2016_flickr.lc.norm.tok.en'
            #     src_valid='../description/bpe/val.lc.norm.tok.bpe.de'
            #     tgt_valid='../description/norm_tok_lc/val.lc.norm.tok.en'
            # else:
            src_test ='../description/bpe/test_2016_flickr.lc.norm.tok.bpe.'+language_src
            src_test2 ='../description/bpe/test_2017_flickr.lc.norm.tok.bpe.'+language_src
            src_test3 ='../description/bpe/test_2017_mscoco.lc.norm.tok.bpe.'+language_src
            tgt_test='../description/norm_tok_lc/test_2016_flickr.lc.norm.tok.'+language_tgt
            tgt_test2 ='../description/norm_tok_lc/test_2017_flickr.lc.norm.tok.'+language_tgt
            tgt_test3 ='../description/norm_tok_lc/test_2017_mscoco.lc.norm.tok.'+language_tgt
            src_valid='../description/bpe/val.lc.norm.tok.bpe.'+language_src
            tgt_valid='../description/norm_tok_lc/val.lc.norm.tok.'+language_tgt
              
            path_to_test_img_feats=' '
            path_to_valid_img_feats=''
            if opt.image_feat_type is not None:  #测试的时候也需要输入图片特征路径
                if str(opt.path_to_valid_img_feats).__contains__('vgg19'):
                    cnn='vgg19_bn'
                elif str(opt.path_to_valid_img_feats).__contains__('ResNets50'):
                    cnn='ResNets50'
                path_to_test_img_feats='-path_to_test_img_feats ' +'../test_2016_flickr-resnet50-res4frelu_cnn_features.hdf5'
                path_to_test_img_feats_2='-path_to_test_img_feats ' +'../test_2017_flickr-resnet50-res4frelu_cnn_features.hdf5'
                path_to_test_img_feats_3='-path_to_test_img_feats ' +'../test_2017_coco-resnet50-res4frelu_cnn_features.hdf5'
                path_to_valid_img_feats='-path_to_test_img_feats '+'../val-resnet50-res4frelu_cnn_features.hdf5'
                
            import subprocess
            cur_path = os.getcwd()
            print("MODEL_SNAPSHOT=%s    && python translate.py -beam_size %s -src %s  -model ${MODEL_SNAPSHOT}  %s -output ${MODEL_SNAPSHOT}.translations-valid -gpu %s   && model=${MODEL_SNAPSHOT}.translations-valid    && cat ${model} | sed -r 's/(@@ )|(@@ ?$)//g' | cat > ${model}.txt &&        cd ../multeval-0.5.1/ &&  cd ../multeval-0.5.1/ &&  ./multeval.sh eval --refs %s \
                --hyps-baseline %s/${model}.txt --meteor.language %s 2>log.txt  && cd %s" % (model_name, opt.beam_size, src_valid,path_to_valid_img_feats,opt.gpuid[0],tgt_valid,cur_path,language_tgt,cur_path))
            
            
            res = subprocess.check_output("MODEL_SNAPSHOT=%s    && python translate.py -beam_size %s -src %s  -model ${MODEL_SNAPSHOT}  %s -output ${MODEL_SNAPSHOT}.translations-valid -gpu %s   && model=${MODEL_SNAPSHOT}.translations-valid    && cat ${model} | sed -r 's/(@@ )|(@@ ?$)//g' | cat > ${model}.txt &&       cd ../multeval-0.5.1/ &&  ./multeval.sh eval --refs %s  --hyps-baseline %s/${model}.txt --meteor.language %s 2>log.txt  && cd %s" % (model_name,opt.beam_size, src_valid,path_to_valid_img_feats,opt.gpuid[0],tgt_valid,cur_path,language_tgt,cur_path),shell=True).decode("utf-8")
            
            content=res.strip().split('\n')
            res=re.sub('\s+',' ', content[-2])
            print('result',content[-3])
            print('result',content[-2])
            bleu=float(res.split()[1])
            # bleu=float(re.sub('.*.*BLEU = ','',res).split(',')[0])
            # print('validation BlEU score',bleu)
            if bleu > best_BLEU:
                best_BLEU=bleu
                best_model_name=model_name
            # print('cur epoch',epoch)
            # print('opt.epochs',opt.epochs)
            if epoch==opt.epochs:
                print("MODEL_SNAPSHOT=%s    && python translate.py -beam_size %s -src %s  -model ${MODEL_SNAPSHOT}  %s -output ${MODEL_SNAPSHOT}.translations-valid -gpu %s   && model=${MODEL_SNAPSHOT}.translations-valid    && cat ${model} | sed -r 's/(@@ )|(@@ ?$)//g' | cat > ${model}.txt &&         cd ../multeval-0.5.1/ &&  ./multeval.sh eval --refs %s  --hyps-baseline %s/${model}.txt --meteor.language %s 2>log.txt  && cd %s" % (best_model_name, opt.beam_size, src_test,path_to_test_img_feats,opt.gpuid[0],tgt_test,cur_path,language_tgt,cur_path))

                outfile.write('best validation BLEU:'+str(best_BLEU)+'\n')
                print('result best validation BLEU:',best_BLEU)

                print('\n\n','*'*5,opt.dropout,' ',opt.embedding_dropout,'*'*5)
                #test 2016
                print('result','*'*5+'  test 2016 '+'*'*5)
                res = subprocess.check_output("MODEL_SNAPSHOT=%s    && python translate.py -beam_size %s -src %s  -model ${MODEL_SNAPSHOT}  %s -output ${MODEL_SNAPSHOT}.translations-test2016 -gpu %s   && model=${MODEL_SNAPSHOT}.translations-test2016    && cat ${model} | sed -r 's/(@@ )|(@@ ?$)//g' | cat > ${model}.txt &&         cd ../multeval-0.5.1/ &&  ./multeval.sh eval --refs %s  --hyps-baseline %s/${model}.txt --meteor.language %s 2>log.txt  && cd %s" % (best_model_name, opt.beam_size, src_test,path_to_test_img_feats,opt.gpuid[0],tgt_test,cur_path,language_tgt,cur_path),shell=True).decode("utf-8")
                content=res.strip().split('\n')
                print('result',content[-3])
                print('result',content[-2])


                #test2017
                print('\n\nresult'+'*'*5+'  test 2017 '+'*'*5)
                res = subprocess.check_output("MODEL_SNAPSHOT=%s    && python translate.py -beam_size %s -src %s  -model ${MODEL_SNAPSHOT}  %s -output ${MODEL_SNAPSHOT}.translations-test2017 -gpu %s   && model=${MODEL_SNAPSHOT}.translations-test2017    && cat ${model} | sed -r 's/(@@ )|(@@ ?$)//g' | cat > ${model}.txt &&         cd ../multeval-0.5.1/ &&  ./multeval.sh eval --refs %s  --hyps-baseline %s/${model}.txt --meteor.language %s 2>log.txt  && cd %s" % (best_model_name, opt.beam_size, src_test2,path_to_test_img_feats,opt.gpuid[0],tgt_test2,cur_path,language_tgt,cur_path),shell=True).decode("utf-8")
                content=res.strip().split('\n')
                print('result',content[-3])
                print('result',content[-2])

                #test coco 2017
                print('\n\nresult'+'*'*5+'  test coco '+'*'*5)
                res = subprocess.check_output("MODEL_SNAPSHOT=%s    && python translate.py -beam_size %s -src %s  -model ${MODEL_SNAPSHOT}  %s -output ${MODEL_SNAPSHOT}.translations-test_coco -gpu %s   && model=${MODEL_SNAPSHOT}.translations-test_coco    && cat ${model} | sed -r 's/(@@ )|(@@ ?$)//g' | cat > ${model}.txt &&         cd ../multeval-0.5.1/ &&  ./multeval.sh eval --refs %s  --hyps-baseline %s/${model}.txt --meteor.language %s 2>log.txt  && cd %s" % (best_model_name, opt.beam_size, src_test3,path_to_test_img_feats,opt.gpuid[0],tgt_test3,cur_path,language_tgt,cur_path),shell=True).decode("utf-8")
                content=res.strip().split('\n')
                print('result',content[-3])
                print('result',content[-2])


            if len(model_name_list)-cnt>3:
                for j in range(len(model_name_list)):
                    if model_name_list[j]!=best_model_name and  os.path.exists(model_name_list[j]):
                        os.remove(model_name_list[j])
                        cnt += 1
                        break

def  check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint):
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
    # if opt.train_from:
    #     print('Loading optimizer from checkpoint.')
    #     optim = checkpoint['optim']
    #     optim.optimizer.load_state_dict(
    #         checkpoint['optim'].optimizer.state_dict())
    # else:
    print('Making optimizer for training.')
    optim = onmt.Optim(
        opt.optim, opt.learning_rate, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        beta1=opt.adam_beta1,
        beta2=opt.adam_beta2,
        adagrad_accum=opt.adagrad_accumulator_init,
        decay_method=opt.decay_method,
        warmup_steps=opt.warmup_steps,
        model_size=opt.rnn_size)

    optim.set_parameters(model.named_parameters())

    return optim


def main():
    train_img_feats=None
    valid_img_feats=None
    if opt.image_feat_type is not None:
        # print('@@@ Train  loading image_feat')
        #加载图片特征
        train_file = tables.open_file(opt.path_to_train_img_feats, mode='r')
        valid_file = tables.open_file(opt.path_to_valid_img_feats, mode='r')

        #直觉上应该应该用local feature 否则attention怎么派上用场呢？
        if opt.image_feat_type == 'local':
            # load only the local image features
            train_img_feats = train_file.root.local_feats#[:]
            valid_img_feats = valid_file.root.local_feats#[:]
        elif opt.image_feat_type == 'global':
            # load only the global image features
            train_img_feats = train_file.root.global_feats
            valid_img_feats = valid_file.root.global_feats
        else:
            raise AssertionError('image_feat_type should be local  or global  or None !')

        # close hdf5 file handlers
        # train_file.close()
        # valid_file.close()

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = 0 #checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train"))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = load_fields(first_dataset, data_type, checkpoint)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_model(model, fields, optim, data_type, model_opt,
                train_img_feats=train_img_feats,
                valid_img_feats=valid_img_feats)


    # If using tensorboard for logging, close the writer after training.
    if opt.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
    print('*'*20,torch_seed,'*'*20)
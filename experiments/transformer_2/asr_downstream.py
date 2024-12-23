import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import re
import time
import yaml
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
import json
# from optim import Optimizer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from transformers import RobertaTokenizer, BertTokenizer, AutoModelForMaskedLM
from transformers import AdamW
import matplotlib.pyplot as plt
from m2p_mask import process_test_MAM_data
from downstream_for_CTAL_asr import RobertaM2Upstream, ASR

import editdistance as ed
# from calculate_eer import get_eer
import Levenshtein
from datetime import date


def _save_canvas(data, meta=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    if meta is None:
        # ax.imshow(data, aspect="auto", origin="lower")  #李宏毅老师原来的版本
        ax.imshow(data, aspect="auto", origin="lower")#横轴代表的是文本每个字的位置，纵轴表示的是语音时间步，使用data.T进行调换
    else:
        ax.bar(meta[0], data[0], tick_label=meta[1], fc=(0, 0, 1, 0.5))
        ax.bar(meta[0], data[1], tick_label=meta[1], fc=(1, 0, 0, 0.5))
    fig.canvas.draw()
    # Note : torch tb add_image takes color as [0,1]
    data = np.array(fig.canvas.renderer._renderer)[:, :, :-1] / 255.0
    plt.close(fig)
    return data


def feat_to_fig(feat):
    # feat TxD tensor
    data = _save_canvas(feat.numpy())
    return torch.FloatTensor(data), "HWC"


def verbose(msg):
    ''' Verbose function for print information to stdout'''
    if type(msg) == list:
        for m in msg:
            print('[INFO]', m.ljust(100))
    else:
        print('[INFO]', msg.ljust(100))


def compute_cer(refs, hypos):
    total_dist, total_len = 0, 0
    for ref, hypo in zip(refs, hypos):
        ref, hypo = ''.join(ref), ''.join(hypo)
        dist = Levenshtein.distance(ref, hypo)
        total_dist += dist
        total_len += len(ref)
    cer = float(total_dist) / float(total_len)
    return cer


def cal_er(tokenizer, pred, truth):
    # Calculate error rate of a batch
    if pred is None:
        return np.nan
    elif len(pred.shape) >= 3:
        pred = pred.argmax(dim=-1)
    er = []
    for p, t in zip(pred, truth):
        p = tokenizer.decode(p.tolist())
        t = tokenizer.decode(t.tolist())
        #过滤掉特殊字符在计算CER
        p = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', p).strip()
        t = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', t).strip()
        er.append(float(ed.eval(p, t)) / len(t))
    return sum(er) / len(er)
def cal_ser(tokenizer, pred, truth):
    # Calculate sentence error rate of a batch
    if pred is None:
        return np.nan
    elif len(pred.shape) >= 3:
        pred = pred.argmax(dim=-1)
    ser = []
    for p, t in zip(pred, truth):
        p = tokenizer.decode(p.tolist())
        t = tokenizer.decode(t.tolist())
        # 过滤掉特殊字符在计算SER
        p = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', p).strip()
        t = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', t).strip()
        if p != t: # 判断预测结果与真实结果是否相同
            ser.append(1) # 句子错误，添加到错误列表
    ser_rate = len(ser) / len(pred) # 计算句子错误率
    return ser_rate
    
def human_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3.1f}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])


def progress(msg, step):
    ''' Verbose function for updating progress on stdout (do not include newline) '''
    sys.stdout.write("\033[K")  # Clear line
    verbose('[{}] {}'.format(human_format(step), msg))

class DownstreamDataset(object):
    def __init__(self, data_list, tokenizer, audio_length=None):
        # gst change:
        self.data_list = data_list
        self.audio_length = audio_length
        self.tokenizer = tokenizer
        self.max_length = 256

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, label = self.data_list[index]
        # audio_path = self.audio_path[index]
        # asr_text = self.asr_text[index]
        # label = self.label[index]
        ##audio_name:原来.csv中是.npy后缀的名字，要获取音频文件需要将.npy替换为.wav
        audio_name = re.sub('\.npy', '.wav', os.path.basename(audio_path))
        audio_input = torch.FloatTensor(np.load(audio_path))
        if self.audio_length is not None: audio_input = audio_input[:self.audio_length, :]
        # Here sometimes the asr input could be the sepearte file path
        # The following preprocess could be modified based on the formats of the text record
        if os.path.isfile(asr_text):
            # The following preprocess could be modified based on the formats of the text record
            asr_text = ' '.join([x.strip('\n').split(',')[0] for x in open(asr_text, 'r').readlines()])

        text_input = self.tokenizer(' '.join(asr_text))

        # print("text_input",text_input)
        # print("===============Dataset,asr_text through tokenizer to text_input: ", len(text_input['input_ids']))

        #######gst change:2.27###############
        label = ' '.join([x.strip('\n').split(',')[0] for x in label])
        # print("add Blank “-” for label：",label)
        label = self.tokenizer(' '.join(label))

        ##########################################################################################
        return {'audio_input': audio_input, 'text_input': text_input, 'label': label, 'audio_name': audio_name}


def collate(sample_list, tokenizer, config):
    batch_audio = [x['audio_input'] for x in sample_list]
    audio_lens = [x['audio_input'].shape[0] for x in sample_list]  # 获取音频序列长度
    pad_batch_audio = pad_sequence(batch_audio, batch_first=True)
    # for x in sample_list:
    #     print("=========text_input_length:",len(x['text_input']['input_ids']))
    #     print("=========audio_name:",x['audio_name'])
    pad_batch_text = {
        'input_ids': [x['text_input']['input_ids'] for x in sample_list],
        'attention_mask': [x['text_input']['attention_mask'] for x in sample_list],
    }
    text_lens = [len(x['text_input']['input_ids']) for x in sample_list]  # 获取文本序列长度
    pad_batch_text = tokenizer.pad(pad_batch_text, return_tensors='pt')
    slc_id = tokenizer.convert_tokens_to_ids('[MASK]')
    non_zero_indices = torch.nonzero(pad_batch_text['input_ids'][:, 1:-1], as_tuple=True)
    pad_batch_text['input_ids'][:, non_zero_indices[1] + 1] = slc_id
    s_inputs = pad_batch_text['input_ids']
    s_attention_mask = pad_batch_text['attention_mask']
    a_attention_mask, a_inputs = process_test_MAM_data((pad_batch_audio,), config)

    labels = [x['label']['input_ids'] for x in sample_list]
    pad_labels = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=0)
    batch_label = pad_labels
    # label_len = batch_label.ne( tokenizer.pad_token_id).sum(dim=-1)
    Batch = batch_label.shape[0]
    fill = batch_label.shape[1]
    label_len = torch.full(size=(Batch,), fill_value=fill, dtype=torch.long)

    audio_lens = torch.tensor(audio_lens)
    text_lens = torch.tensor(text_lens)
    # print("true audio len:", audio_lens)
    # print("true text len:", text_lens)
    ######################################################
    batch_name = [x['audio_name'] for x in sample_list]
    return ((a_inputs, a_attention_mask),
            (s_inputs, s_attention_mask),
            batch_label, batch_name, label_len, audio_lens, text_lens)


def run(args, config, train_data, valid_data):
    ############################ PARAMETER SETTING ##########################
    num_workers = config['dataloader']['n_jobs']
    batch_size = config['dataloader']['batch_size']
    audio_length = 3000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    ############################## PREPARE DATASET ##########################
    train_dataset = DownstreamDataset(train_data, tokenizer, audio_length)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        collate_fn=lambda x: collate(x, tokenizer, config['upstream']['acoustic']),
        shuffle=True, drop_last=True, num_workers=num_workers
    )

    valid_dataset = DownstreamDataset(valid_data, tokenizer, audio_length)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size,
        collate_fn=lambda x: collate(x, tokenizer, config['upstream']['acoustic']),
        shuffle=False, drop_last=True, num_workers=num_workers
    )
    # #
    # if test_data is None:
    #     test_data = valid_data
    # test_dataset = DownstreamDataset(test_data, tokenizer, audio_length)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset, batch_size=batch_size,
    #     collate_fn=lambda x: collate(x, tokenizer, config['upstream']['acoustic']),
    #     shuffle=False, num_workers=num_workers
    # )
    ########################### CREATE MODEL #################################
    # ASREncoder = MultiModalEncoderDecoder(ckpt_path=args.ckpt_path)
    # ASREncoder.cuda()
    # init_adadelta = config['hparas']['optimizer'] == 'Adadelta'
    vocab_size = tokenizer.vocab_size
    ASRModel = ASR(ckpt_path=args.ckpt_path,vocab_size=vocab_size, attention=config['attention'], decoder=config['decoder']).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in ASRModel.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in ASRModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)#orignal T_max=epochs
    # asrmodel_paras = [{'params': ASRModel.parameters()}]
    # model_parameters = ASRModel.parameters()
    # loss
    # ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)  #roignal
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)   #3.26 rewrite
    # optimizer = Optimizer(asrmodel_paras, **config['hparas'])
    ########################### TRAINING #####################################
    count = 0
    att_loss = None
    step = 0
    valid_step = 2192 #one epoch do one valid
    max_step = 109600 #50 epochs
    # 初始化最小CER值为一个很大的数
    min_cer = 1000.0
    min_ser = 1000.0
    # 初始化最小CER值对应的模型参数
    best_model_params = None
    # 初始化所有验证平均CER值的列表
    all_cer_values = []
    n_epochs = 0

    valid_log_file = '/vol/nfs/mgl/GST/3_31_ctaL/CTAL-main/Ablation_Study_LK_Pretrained/transformer_layer_2/decode_log'
    if not os.path.exists(valid_log_file):
        open(valid_log_file, 'a').close()
    ori_logdir = args.logdir
    today = date.today()
    logdir = os.path.join(ori_logdir, 'train-{}'.format(today))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log = SummaryWriter(logdir, flush_secs=180)
    while step < max_step:
        # progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        # progress = tqdm(train_loader, desc='Epoch:{}'.format(epoch+1))
        ASRModel.train()
        for acoustic_inputs, semantic_inputs, label_inputs, _, label_len, audio_lens, text_lens in train_loader:

            speech_inputs = acoustic_inputs[0].to(device)
            speech_attention_mask = acoustic_inputs[1].to(device)
            text_inputs = semantic_inputs[0].to(device)
            text_attention_mask = semantic_inputs[1].to(device)
            # print("train_textinput:",text_inputs)
            label_inputs = label_inputs.to(device)
            # 计算label标签的长度，padding上的（补0）的不计数,转为tensor
            # label_len = torch.sum(label_inputs != 0, dim=-1)
            #####gst change:2.26#####
            # label_len = torch.count_nonzero(label_inputs != 0,dim=1)
            # tf_rate = optimizer.pre_step(step)####3.18####

            total_loss = 0
            # real_len, att_output, att_align, dec_state = ASRModel(text_inputs, text_attention_mask,
            #                                                       speech_inputs, speech_attention_mask,
            #                                                       audio_lens, batch_size,
            #                                                       label_inputs.size(-1), tf_rate=1,
            #                                                  teacher=label_inputs, get_dec_state=False)

            real_len, att_output, att_align, dec_state= ASRModel(text_inputs=text_inputs,
                                                                  text_attention_mask=text_attention_mask,
                                                                  speech_inputs=speech_inputs,
                                                                  speech_attention_mask=speech_attention_mask,
                                                                  real_len=audio_lens,
                                                                  batch_size=batch_size,
                                                                  decode_step=label_inputs.size(
                                                                      -1), tf_rate=1,
                                                                  teacher=label_inputs,
                                                                  get_dec_state=False)
            if att_output is not None:
                b, t, _ = att_output.shape
                att_loss = ce_loss(att_output.view(b * t, -1), label_inputs.view(-1))
                total_loss += att_loss

            # grad_norm = backward(total_loss, model_parameters, optimizer, step)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ASRModel.parameters(), 5)
            optimizer.step()
            scheduler.step()
            # optimizer.step()
            step += 1
            # ori_logdir = args.logdir
            # today = date.today()
            # logdir = os.path.join(ori_logdir,'train-{}'.format(today))
            # if not os.path.exists(logdir):
            #     os.makedirs(logdir)
            # log = SummaryWriter(logdir, flush_secs=180)

            if (step == 1) or (step % 100 == 0):
                progress('Tr stat | Loss - {:.6f} | Grad. Norm - {:.2f}'
                         .format(total_loss.cpu().item(), grad_norm), step)
                log_dict = {'tr_att': att_loss}
                log_name = 'Train_Loss'
                if type(log_dict) is dict:
                    log_dict = {key: val for key, val in log_dict.items() if (
                            val is not None and not math.isnan(val))}
                if log_dict is None:
                    pass
                elif len(log_dict) > 0:
                    log.add_scalars(log_name, log_dict, step)
                log_name_cer = 'Train_CER'
                log_dict_cer = {'tr_att': cal_er(tokenizer, att_output, label_inputs)}
                if type(log_dict_cer) is dict:
                    log_dict_cer = {key: val for key, val in log_dict_cer.items() if (
                            val is not None and not math.isnan(val))}
                if log_dict_cer is None:
                    pass
                elif len(log_dict_cer) > 0:
                    log.add_scalars(log_name_cer, log_dict_cer, step)
                ####################################################
            if (step == 1) or (step % valid_step == 0):
                # Eval mode
                ASRModel.eval()
                dev_cer = []
                dev_ser = []
                time.sleep(2)
                start_log_index = max(len(valid_loader) - 2, 0)
                for i, (
                        acoustic_inputs, semantic_inputs, label_inputs, _, label_len, audio_lens,
                        text_lens) in enumerate(
                    valid_loader):
                    # progress('Valid step - {}/{}'.format(i + 1, len(valid_loader)))
                    speech_inputs = acoustic_inputs[0].to(device)
                    speech_attention_mask = acoustic_inputs[1].to(device)
                    text_inputs = semantic_inputs[0].to(device)
                    # print("valid_textinput:", text_inputs)
                    text_attention_mask = semantic_inputs[1].to(device)
                    label_inputs = label_inputs.to(device)
                    # Forward model
                    with torch.no_grad():
                        DEV_STEP_RATIO = 1.2
                        # real_len, att_output, att_align, dec_state = ASRModel(text_inputs, text_attention_mask,
                        #                                                       speech_inputs, speech_attention_mask,
                        #                                                       audio_lens, batch_size,
                        #                                                       int(label_inputs.size(-1)),
                        #                                                       get_dec_state=False)
                        real_len, att_output, att_align, dec_state= ASRModel(text_inputs=text_inputs,
                                                                                    text_attention_mask=text_attention_mask,
                                                                                    speech_inputs=speech_inputs,
                                                                                    speech_attention_mask=speech_attention_mask,
                                                                                    real_len=audio_lens,
                                                                                    batch_size=batch_size,
                                                                                    decode_step=int(
                                                                                        label_inputs.size(-1)),
                                                                                    get_dec_state=False)
                    dev_cer.append(cal_er(tokenizer, att_output, label_inputs))
                    dev_ser.append(cal_ser(tokenizer, att_output, label_inputs))
                    log_name_valcer = 'Val_CER'
                    log_dict_valcer = {'val_att': cal_er(tokenizer, att_output, label_inputs)}
                    if type(log_dict_valcer) is dict:
                        log_dict_valcer = {key: val for key, val in log_dict_valcer.items() if (
                                val is not None and not math.isnan(val))}
                    if log_dict_valcer is None:
                        pass
                    elif len(log_dict_valcer) > 0:
                        log.add_scalars(log_name_valcer, log_dict_valcer, step)
                    # Show some example on tensorboard
                    for j in label_inputs:
                        true_text = tokenizer.decode(j.tolist())
                        true_text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', true_text).strip()
                        with open(valid_log_file, 'a') as f:
                            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f.write(f'[{time_str}][{"In Valid-loader:"}{i}][{"True-text:"}]{true_text}\n')
                            # f.write(f'')
                            # f.write(f'{"True-text:"}{true_text}\n')
                            f.close()

                    for p in att_output:
                        attn_text = tokenizer.decode(p.argmax(dim=-1).tolist())
                        # Filter out special tokens
                        attn_text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', attn_text).strip()
                        with open(valid_log_file, 'a') as f:
                            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # f.write(f'[{time_str}]')
                            f.write(f'[{time_str}][{"In Valid-loader:"}{i}][{"Pred-text:"}]{attn_text}\n')
                            f.close()
                    # if i == len(valid_loader) // 2:
                    if i >= start_log_index:
                        for j in range(batch_size):
                            log_name_val = 'True_text{}'.format(j)
                            true_text = tokenizer.decode(label_inputs[j].tolist())
                            log_dict_val = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', true_text).strip()
                            # if type(log_dict_val) is dict:
                            #     log_dict_val = {key: val for key, val in log_dict_val.items() if (
                            #             val is not None and not math.isnan(val))}
                            # if log_dict_val is None:
                            #     pass
                            if len(log_dict_val) > 0:
                                log.add_text(log_name_val, log_dict_val, step)

                            log_name_align = 'att_align{}'.format(i)
                            log_dict_align = feat_to_fig(
                                att_align[j, 0, :, :].cpu().detach())
                            img, form = log_dict_align
                            # if type(log_dict_align) is dict:
                            #     log_dict_align = {key: val for key, val in log_dict_align.items() if (
                            #             val is not None and not math.isnan(val))}
                            # if log_dict_align is None:
                            #     pass
                            if len(log_dict_align) > 0:
                                log.add_image(log_name_align, img, global_step=step, dataformats=form)
                            log_name_pred = 'Pred_text{}'.format(j)
                            pred_text = tokenizer.decode(att_output[j].argmax(dim=-1).tolist())
                            log_dict_pred = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', pred_text).strip()
                            # if type(log_dict_pred) is dict:
                            #     log_dict_pred = {key: val for key, val in log_dict_pred.items() if (
                            #             val is not None and not math.isnan(val))}
                            # if log_dict_pred is None:
                            #     pass
                            if len(log_dict_pred) > 0:
                                log.add_text(log_name_pred, log_dict_pred, step)
                # 计算当前验证的平均CER值
                avg_cer = sum(dev_cer) / len(dev_cer)
                avg_ser = sum(dev_ser) / len(dev_ser)
                # 记录所有验证的平均CER值
                all_cer_values.append(avg_cer)
                # 如果当前CER值比之前的最小值小，就更新最小CER值和对应的模型参数
                if avg_cer < min_cer:
                    min_cer = avg_cer
                    # 保存当前最佳模型参数到文件中
                    ckppdir = '/vol/nfs/mgl/GST/3_31_ctaL/CTAL-main/Ablation_Study_LK_Pretrained/transformer_layer_2/result'
                    ckpt_path = os.path.join(ckppdir, "best_att.pth")
                    full_dict = {
                        "model": ASRModel.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": step,
                        'CER': min_cer
                    }
                    torch.save(full_dict, ckpt_path)
                    verbose("Saved checkpoint (step = {}, CER = {:.6f}) and status @ {}".format(human_format(step),
                                                                                                min_cer, ckpt_path))                                                                      
                if avg_ser < min_ser:
                    min_ser = avg_ser
                    verbose("Saved checkpoint (step = {}, CER = {:.6f})".format(human_format(step),
                                                                                                min_ser))
                # Resume training
                ASRModel.train()

            torch.cuda.empty_cache()
            # timer = Timer()
            # timer.set()
            if step > max_step:
                break
        n_epochs += 1

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default=None, help='downstream task name')
    parser.add_argument("--config", type=str, default=None, help='configuration file path')
    parser.add_argument("--ckpt_path", type=str, default=None, help='checkpoint file path')
    parser.add_argument("--tokenizer_path", type=str, default=None, help='pretrained tokenizer file path')
    # parser.add_argument("--epochs", type=int, default=20, help="training epoches")
    parser.add_argument("--save_path", type=str, default='./save_asr_result',
                        help="report or ckpt save path")  # default=None
    parser.add_argument("--freeze", type=bool, default=False, help="freeze the pretrain model")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument('--logdir', default='log/', type=str,help='Logging path.', required=False)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    report_result = []
    #
    train_data_root = '/vol/nfs/mgl/GST/3_31_ctaL/CTAL-main/dataset/20h_parallel'
    # gst change 2.27(将train_data,valid_data,test_data传入):
    train_data_table = '/vol/nfs/mgl/GST/3_31_ctaL/CTAL-main/dataset/20h_parallel_for_finntune.csv'
    train_tables = [pd.read_csv(train_data_table)]
    train_tables = pd.concat(train_tables, ignore_index=True).sort_values(by=['length'], ascending=False)
    train_audio_path = train_tables['file_path'].tolist()
    train_asr_text = train_tables['align_path'].tolist()
    train_label = train_tables['label'].tolist()
    # valid_data:
    valid_data_root = '/vol/nfs/mgl/GST/3_31_ctaL/CTAL-main/dataset/downstream_val'
    valid_data_table = '/vol/nfs/mgl/GST/3_31_ctaL/CTAL-main/dataset/downstream_val.csv'
    valid_tables = [pd.read_csv(valid_data_table)]
    valid_tables = pd.concat(valid_tables, ignore_index=True)
    valid_audio_path = valid_tables['file_path'].tolist()
    valid_asr_text = valid_tables['align_path'].tolist()
    valid_label = valid_tables['label'].tolist()

    train_data = list(zip(train_audio_path, train_asr_text, train_label))
    valid_data = list(zip(valid_audio_path, valid_asr_text, valid_label))
    # test_data = list(zip(test_audio_path, test_asr_text, test_label))

    # print(train_data[0])
    # data_table = './dataset/finetune_test.csv'
    r = run(args, config, train_data, valid_data)
    # report_result = [r]

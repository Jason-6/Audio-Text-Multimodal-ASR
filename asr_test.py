import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import re
import time
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import copy
import torch
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
import math
from m2p_mask import process_test_MAM_data
from downstream_for_CTAL_asr import RobertaM2Upstream, ASR
from transformers import BertTokenizer
from transformers import AdamW


def greedy_decode(self, dv_set):
    ''' Greedy Decoding '''
    results = []
    for i, data in enumerate(dv_set):
        self.progress('Valid step - {}/{}'.format(i + 1, len(dv_set)))
        # Fetch data
        feat, feat_len, txt, txt_len = self.fetch_data(data)
        # Forward model
        with torch.no_grad():
            ctc_output, encode_len, att_output, att_align, dec_state = \
                self.decoder(feat, feat_len, int(float(feat_len.max()) * self.config['decode']['max_len_ratio']),
                             emb_decoder=self.emb_decoder)
        for j in range(len(txt)):
            idx = j + self.config['data']['corpus']['batch_size'] * i
            if att_output is not None:
                hyp_seqs = att_output[j].argmax(dim=-1).tolist()
            else:
                hyp_seqs = ctc_output[j].argmax(dim=-1).tolist()
            true_txt = txt[j]
            results.append((str(idx), [hyp_seqs], true_txt))
    return results

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
    #####gst :将文本输入的input_ids和attention_mask均设置###############
    slc_id = tokenizer.convert_tokens_to_ids('[MASK]')
    non_zero_indices = torch.nonzero(pad_batch_text['input_ids'][:, 1:-1], as_tuple=True)
    pad_batch_text['input_ids'][:, non_zero_indices[1] + 1] = slc_id
    s_inputs = pad_batch_text['input_ids']
    #####################################################################
    # slc_id = tokenizer.convert_tokens_to_ids('[MASK]')
    # cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    # sep_id =  tokenizer.convert_tokens_to_ids('[SEP]')
    # pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    # non_special_token = (pad_batch_text['input_ids'] != cls_id) & (pad_batch_text['input_ids'] != sep_id)& (pad_batch_text['input_ids'] != pad_id)
    # # non_zero_indices = torch.nonzero(pad_batch_text['input_ids'][:, 1:-1], as_tuple=True)
    # pad_batch_text['input_ids'][non_special_token] = slc_id
    # s_inputs = pad_batch_text['input_ids']
    ######################################################################
    # cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    # sep_id =  tokenizer.convert_tokens_to_ids('[SEP]')
    # non_special_token = (pad_batch_text['input_ids'] != cls_id) & (pad_batch_text['input_ids'] != sep_id)
    # pad_batch_text['attention_mask'][non_special_token] = 0
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
    # print("batch_name",batch_name)
    #
    return ((a_inputs, a_attention_mask),
            (s_inputs, s_attention_mask),
            batch_label, batch_name, label_len, audio_lens, text_lens)

def run(args, config, test_data):
    num_workers = config['dataloader']['n_jobs']
    batch_size = config['dataloader']['batch_size']
    audio_length = 3000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    test_dataset = DownstreamDataset(test_data, tokenizer, audio_length)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        collate_fn=lambda x: collate(x, tokenizer, config['upstream']['acoustic']),
        shuffle=False, num_workers=num_workers
    )

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
    output_file = str(args.logdir) + '_{}_{}.csv'

    greedy = config['decode']['beam_size'] == 1
    decoder = copy.deepcopy(ASRModel).to(device)




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
    # valid_data:
    test_data_root = '/home/geshuting/dataset/test'
    test_data_table = '/home/geshuting/Code/CTAL-main/CTAL-main/dataset/test.csv'
    test_tables = [pd.read_csv(test_data_table)]
    test_tables = pd.concat(test_tables, ignore_index=True).sort_values(by=['length'], ascending=False)
    test_audio_path = test_tables['file_path'].tolist()
    test_asr_text = test_tables['align_path'].tolist()
    test_label = test_tables['label'].tolist()

    test_data = list(zip(test_audio_path, test_asr_text, test_label))

    # print(train_data[0])
    # data_table = './dataset/finetune_test.csv'
    r = run(args, config, test_data)
    # report_result = [r]

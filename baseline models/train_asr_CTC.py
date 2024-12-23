# -*- coding: utf-8 -*-

import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from glob import glob
import librosa
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
import librosa.display
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
torch.multiprocessing.set_sharing_strategy('file_system')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# 构建模型
class AsrModel:
    def __init__(self, voc_num, feature_num):
        self.voc_num = voc_num
        self.audio_feature = layers.Input(shape=[None, feature_num], name='audio_feature')

        asr_output = self.vgg_model()

        self.model = tf.keras.Model(inputs=self.audio_feature, outputs=asr_output)
        self.model.summary()

    def asr_resnet_model(self):
        input_features = self.audio_feature[:, :, :, tf.newaxis]

        # 第一个卷积层
        x = layers.Conv2D(64, kernel_size=7, strides=2, padding='valid')(input_features)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)

        # 残差块
        x = self.resnet_block(x, filters=64, kernel_size=3, strides=1)
        x = self.resnet_block(x, filters=64, kernel_size=3, strides=1)
        x = self.resnet_block(x, filters=64, kernel_size=3, strides=1)

        x = self.resnet_block(x, filters=64, kernel_size=3, strides=1)
        x = self.resnet_block(x, filters=64, kernel_size=3, strides=1)
        x = self.resnet_block(x, filters=64, kernel_size=3, strides=1)

        x = layers.Reshape([-1, x.shape[2] * x.shape[3]])(x)
        x = layers.ReLU()(x)

        x = layers.Dense(self.voc_num)(x)  # 全连接层
        x = layers.Activation('softmax', name='Activation0')(x)

        return x

    def resnet_block(self, x, filters, kernel_size, strides):
        residual = x
        x = layers.Conv2D(filters, kernel_size, strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = x + residual
        x = layers.ReLU()(x)
        return x

    def cldnn_model(self):  # 同上
        input_features = self.audio_feature[:, :, :, tf.newaxis]
        x = layers.Conv2D(32, kernel_size=[3, 3], padding='same')(input_features)
        x = layers.BatchNormalization(epsilon=1e-5)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D((2, 2))(x)

        x = layers.Conv2D(64, kernel_size=[3, 3], padding='same')(x)
        x = layers.BatchNormalization(epsilon=1e-5)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D((2, 2))(x)

        x = layers.Reshape([-1, x.shape[2] * x.shape[3]])(x)

        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Dense(512)(x)
        x = layers.Dense(self.voc_num)(x)

        x = layers.Softmax(name='ctc_output')(x)

        return x

    def mrcldnn_model(self,need_lstm = False):  # 同上
        input_features = self.audio_feature[:, :, :, tf.newaxis]
        x1 = layers.Conv2D(16, kernel_size=[3, 3], padding='same')(input_features)
        x1 = layers.BatchNormalization(epsilon=1e-5)(x1)
        x1 = layers.ReLU()(x1)
        x1 = layers.MaxPool2D((2, 2))(x1)

        x1 = layers.Conv2D(32, kernel_size=[3, 3], padding='same')(x1)
        x1 = layers.BatchNormalization(epsilon=1e-5)(x1)
        x1 = layers.ReLU()(x1)
        x1 = layers.MaxPool2D((2, 2))(x1)

        x1 = layers.Conv2D(64, kernel_size=[3, 3], padding='same')(x1)
        x1 = layers.BatchNormalization(epsilon=1e-5)(x1)
        x1 = layers.ReLU()(x1)
        x1 = layers.MaxPool2D((2, 2))(x1)

        x2 = layers.Conv2D(16, kernel_size=[5, 5], padding='same')(input_features)
        x2 = layers.BatchNormalization(epsilon=1e-5)(x2)
        x2 = layers.ReLU()(x2)
        x2 = layers.MaxPool2D((2, 2))(x2)

        x2 = layers.Conv2D(32, kernel_size=[5, 5], padding='same')(x2)
        x2 = layers.BatchNormalization(epsilon=1e-5)(x2)
        x2 = layers.ReLU()(x2)
        x2 = layers.MaxPool2D((2, 2))(x2)

        x2 = layers.Conv2D(64, kernel_size=[5, 5], padding='same')(x2)
        x2 = layers.BatchNormalization(epsilon=1e-5)(x2)
        x2 = layers.ReLU()(x2)
        x2 = layers.MaxPool2D((2, 2))(x2)

        x3 = layers.Conv2D(16, kernel_size=[7, 7], padding='same')(input_features)
        x3 = layers.BatchNormalization(epsilon=1e-5)(x3)
        x3 = layers.ReLU()(x3)
        x3 = layers.MaxPool2D((2, 2))(x3)

        x3 = layers.Conv2D(32, kernel_size=[7, 7], padding='same')(x3)
        x3 = layers.BatchNormalization(epsilon=1e-5)(x3)
        x3 = layers.ReLU()(x3)
        x3 = layers.MaxPool2D((2, 2))(x3)

        x3 = layers.Conv2D(64, kernel_size=[7, 7], padding='same')(x3)
        x3 = layers.BatchNormalization(epsilon=1e-5)(x3)
        x3 = layers.ReLU()(x3)
        x3 = layers.MaxPool2D((2, 2))(x3)

        x = x1 + x2 + x3
        x = layers.Reshape([-1, x.shape[2] * x.shape[3]])(x)

        x = layers.Dense(512)(x)
        if need_lstm:
            x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
            x = layers.Dense(512)(x)
        x = layers.Dense(self.voc_num)(x)

        x = layers.Softmax(name='ctc_output')(x)

        return x

    def vgg_model(self):
        input_features = self.audio_feature[:, :, :, tf.newaxis]
        x = layers.Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv0')(input_features)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN0')(x)
        x = layers.Activation('relu', name='Act0')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv1')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN1')(x)
        x = layers.Activation('relu', name='Act1')(x)
        x = layers.MaxPooling2D(pool_size=2, strides=None, padding="valid")(x)  # 池化层
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv2')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN2')(x)
        x = layers.Activation('relu', name='Act2')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv3')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN3')(x)
        x = layers.Activation('relu', name='Act3')(x)
        x = layers.MaxPooling2D(pool_size=2, strides=None, padding="valid")(x)  # 池化层
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv4')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN4')(x)
        x = layers.Activation('relu', name='Act4')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv5')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN5')(x)
        x = layers.Activation('relu', name='Act5')(x)
        x = layers.MaxPooling2D(pool_size=2, strides=None, padding="valid")(x)  # 池化层
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv6')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN6')(x)
        x = layers.Activation('relu', name='Act6')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv7')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN7')(x)
        x = layers.Activation('relu', name='Act7')(x)
        x = layers.MaxPooling2D(pool_size=1, strides=None, padding="valid")(x)  # 池化层
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv8')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN8')(x)
        x = layers.Activation('relu', name='Act8')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal',
                                     name='Conv9')(x)  # 卷积层
        x = layers.BatchNormalization(epsilon=0.0002, name='BN9')(x)
        x = layers.Activation('relu', name='Act9')(x)
        x = layers.MaxPooling2D(pool_size=1, strides=None, padding="valid")(x)  # 池化层
        x = layers.Dropout(0.2)(x)

        x = layers.Reshape((-1, x.shape[2] * x.shape[3]), name='Reshape0')(
            x)  # Reshape层

        x = layers.Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal',
                                    name='Dense0')(
            x)  # 全连接层

        x = layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)

        x = layers.Dense(self.voc_num, use_bias=True, kernel_initializer='he_normal', name='Dense1')(
            x)  # 全连接层
        x = layers.Activation('softmax', name='Activation0')(x)

        return x

class AudioDataset(Dataset):  # 加载数据
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        wav_path = self.train_data[index][0]
        sentence = self.train_data[index][1]
        # txt_path = self.train_data[index][1]
        # txt_path = txt_path.replace('train', 'data').replace('test', 'data')
        # with open(txt_path, 'r') as f:  # 获取句子
        #     sentence = f.readlines()[0]
        #     sentence = sentence.replace('\n', '').replace(' ', '')

        sentence = sentence.replace('\n', '').replace(' ', '')
        label_id = np.array(self.convert_text(sentence))  # 转为label_id
        wav_feature = self.extract_fbank(wav_path)

        return wav_feature, label_id

    def convert_text(self, text):  # 将文字转为对应的id
        text = text.replace(' ', '')
        ids = []
        for char in text:
            if char in char2idx:
                ids.append(char2idx[char])
            else:
                ids.append(unk_id)
        return ids

    def extract_fbank(self, wav_path, n_mels=80):  # 提取logfbank信息
        sr = 16000
        hop_length = 160
        win_length = 400
        nfft = 512

        samples = librosa.load(wav_path, sr=sr)[0]

        fbank = librosa.feature.melspectrogram(y=samples, sr=sr, hop_length=hop_length, win_length=win_length,
                                               n_mels=n_mels, n_fft=nfft, htk=True).astype(np.float32)
        fbank = fbank.T

        mel_log = np.clip(fbank, a_min=1e-10, a_max=None)
        mel_log = np.log10(mel_log)
        mel_log = np.maximum(mel_log, mel_log.max() - 8.0)
        mel_log = (mel_log + 4.0) / 4.0

        return mel_log


def collate_fn(data):
    # 长度排序
    len_arr = np.array([ld[0].shape[0] for ld in data])
    idxs = np.argsort(-len_arr)

    audio_features = [data[idx][0] for idx in idxs]
    label_ids = [data[idx][1] for idx in idxs]

    label_lens = np.array([label_id.shape[0] for label_id in label_ids])
    audio_feature_lens = np.array([get_audio_len(wav_feature.shape[0]) for wav_feature in audio_features])

    audio_features = pad_data(audio_features)
    label_ids = pad_data(label_ids)

    return audio_features, audio_feature_lens, label_ids, label_lens


def get_audio_len(audio_len):
    # audio_len = (audio_len - 7) // 2 + 1
    # audio_len = (audio_len - 3) // 2 + 1
    audio_len = (audio_len - 2) // 2 + 1
    audio_len = (audio_len - 2) // 2 + 1
    audio_len = (audio_len - 2) // 2 + 1
    return audio_len

def pad_data(inputs, maxlen=None):  # 进行补零
    PAD = 0

    def pad_zero(x, max_len):
        if len(x.shape) == 1:
            x_padded = np.pad(
                x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD
            )
        else:
            x_padded = np.pad(
                x, ((0, max_len - np.shape(x)[0]), (0, 0)), mode="constant", constant_values=PAD
            )
        return x_padded

    if maxlen is None:
        maxlen = max(np.shape(x)[0] for x in inputs)

    inputs = [pad_zero(x, maxlen) for x in inputs]
    inputs = np.array(inputs)
    return inputs


def compute_ctc_loss(label_ids, ctc_input, label_length, input_mels_length):
    # blank_index默认-1，无法指定。接口里面说不需要softmax，但貌似无效。需要进行softmax操作，模型已经softmax过。
    loss = tf.keras.backend.ctc_batch_cost(label_ids, ctc_input, input_mels_length.reshape([-1, 1]),
                                           label_length.reshape([-1, 1]))
    # loss = tf.nn.ctc_loss(label_ids, ctc_input, label_length.reshape([-1]), input_mels_length.reshape([-1]),
    #                       logits_time_major=False, blank_index=blank_index)
    loss = tf.reduce_mean(loss)
    return loss


def chunks(arr_list, num):
    n = int(math.ceil(len(arr_list) / float(num)))
    return [arr_list[i:i + n] for i in range(0, len(arr_list), n)]


def pre_aishell_data():
    train_paths = glob('/data1/asr/data_aishell/wav/train/*/*.wav')
    test_paths = glob('/data1/asr/data_aishell/wav/test/*/*.wav')
    train_dict = {}
    test_dict = {}
    for tp in train_paths:
        tp_name = os.path.splitext(os.path.split(tp)[1])[0]
        train_dict[tp_name] = tp
    for tp in test_paths:
        tp_name = os.path.splitext(os.path.split(tp)[1])[0]
        test_dict[tp_name] = tp

    train_data, test_data = [], []
    with open('/data1/asr/data_aishell/transcript/aishell_transcript_v0.8.txt', 'r') as fr:
        for line in tqdm(fr.readlines()):
            line = line.replace('\n', '')
            wav_name = line[:16]
            text = line[16:]

            if wav_name in train_dict:
                train_data.append([train_dict[wav_name], text])
            elif wav_name in test_dict:
                test_data.append([test_dict[wav_name], text])
            else:
                continue

    return train_data, test_data


def pre_thchs_data():  # 准备数据
    train_data = []
    wav_paths = sorted(glob('/data1/asr/data_thchs30/train/*.wav'))
    txt_paths = sorted(glob('/data1/asr/data_thchs30/train/*.trn'))
    for i in range(len(wav_paths)):
        train_data.append([wav_paths[i], txt_paths[i]])

    test_data = []
    wav_paths = sorted(glob('/data1/asr/data_thchs30/test/*.wav'))
    txt_paths = sorted(glob('/data1/asr/data_thchs30/test/*.trn'))
    for i in range(len(wav_paths)):
        test_data.append([wav_paths[i], txt_paths[i]])
    return train_data, test_data


def load_frame(path,model_name):  # 加载模型
    frame_names = {}
    for frame_name in glob(f'{path}/frame_{model_name}*'):
        name = os.path.split(frame_name)[1]
        if len(name.split('_')) == 2:
            frame_names[int(name.split('_')[1])] = frame_name

    if len(sorted(frame_names)) == 0:
        return None, None
    else:
        frame_index = sorted(frame_names)[-1]
        return frame_names[frame_index], frame_index


def delete_frame(path,model_name):  # 删除模型
    frame_names = {}
    for frame_name in glob(f'{path}/frame_{model_name}*'):
        name = os.path.split(frame_name)[1]
        if len(name.split('_')) == 2:
            frame_names[int(name.split('_')[1])] = frame_name

    for delete_key in sorted(frame_names)[:-5]:
        os.remove(frame_names[delete_key])


def beam_search(y, beam_size=10):
    # y是个二维数组，记录了所有时刻的所有项的概率
    T = y.shape[0]
    # p1p2...pn改为log(p1p2...pn)=logp1+logp2+...+logpn。
    log_y = np.log(y)
    # 初始的beam
    beam = [([], 0)]
    # 遍历所有时刻t
    for t in range(T):
        # 每个时刻先初始化一个new_beam
        new_beam = []
        # 遍历beam
        for prefix, score in beam:
            # 先对概率排序，然后选取排序好概率值的一定范围去排序，避免如1e-10这样的小概率进行干扰，如选取前10个最大的
            score_index = np.argsort(log_y[t])[-10:]
            for i in score_index:
                # 记录添加的新项是这个时刻的第几项，对应的概率(log形式的)加上新的这项log形式的概率(本来是乘的，改成log就是加)
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]
                # new_beam记录了对于beam中某一项，将这个项分别加上新的时刻中的每一项后的概率
                new_beam.append((new_prefix, new_score))
        # 给new_beam按score排序
        new_beam.sort(key=lambda x: x[1], reverse=True)
        # beam即为new_beam中概率最大的beam_size个路径
        beam = new_beam[:beam_size]

    return beam


def remove_blank(labels, blank=0):
    new_labels = []
    # 合并相同的标签
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def greedy_decode(y, blank_index=0):  # 贪婪解码
    # 按列取最大值，即每个时刻t上最大值对应的下标
    raw_rs = np.argmax(y, axis=1)
    # 移除blank,值为0的位置表示这个位置是blank
    rs = remove_blank(raw_rs, blank_index)
    return raw_rs, rs


# 编辑距离
def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str(str1[i - 1]) == str(str2[j - 1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


if __name__ == '__main__':
    # 参数
    batch_size = 64
    epoch = 20
    initial_learning_rate = 6e-4
    decay_steps = 5000
    decay_rate = 0.95
    feature_num = 80
    model_name = 'vgg'
    pb_path = f'./resources/{model_name}_model'

    # 准备数据
    vocab = []
    with open('resources/vocab_4264.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            vocab.append(line)
    vocab.append('unk')
    vocab.append('blank_')
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    char_num = len(vocab)
    unk_id = char_num - 2
    blank_index = char_num - 1

    ''' ----------------------------加载模型------------------------------- '''
    if not os.path.exists(pb_path):
        os.makedirs(pb_path)

    asr_train_model = AsrModel(char_num, feature_num).model
    # 恢复权重
    frame_path, frame_index = load_frame(pb_path,model_name)
    if frame_path is None:
        last_epoch = -1
        epoch_start = 1
    else:
        asr_train_model.load_weights(frame_path)
        last_epoch = frame_index
        epoch_start = last_epoch + 1

        print(f'恢复权重 = {frame_path}')

    ''' ------------------------准备数据------------------------------ '''
    train_data, test_data = pre_aishell_data()

    optimizer = tf.keras.optimizers.Adam(  # 优化器
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                     decay_steps=decay_steps,
                                                                     decay_rate=decay_rate))

    train_loader = DataLoader(dataset=AudioDataset(train_data), batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=AudioDataset(test_data), batch_size=batch_size, shuffle=True, drop_last=False,
                             num_workers=4, collate_fn=collate_fn)
    for epoch_index in range(epoch_start, epoch):
        train_cer_result_list = []
        for data in tqdm(train_loader):
            audio_features, audio_feature_lens, label_ids, label_lens = data

            # 梯度下降
            with tf.GradientTape(persistent=False) as tape:
                asr_output = asr_train_model(audio_features, training=True)
                ctc_loss = compute_ctc_loss(label_ids, asr_output, label_lens, audio_feature_lens)

                grads = tape.gradient(target=ctc_loss, sources=asr_train_model.trainable_variables)  # 梯度优化
                optimizer.apply_gradients(zip(grads, asr_train_model.trainable_variables))

            print(f'epoch = {epoch_index}  total_loss = {ctc_loss}')

            for i in range(len(asr_output)):
                asr_result = asr_output[i][:audio_feature_lens[i]]
                label_id = label_ids[i][:label_lens[i]]

                ctc_output = greedy_decode(asr_result, blank_index)[1]
                cer_result = edit(list(ctc_output), label_id) / len(label_id)

                label_txt = ''.join([idx2char[l] for l in label_id])
                pre_txt = ''.join([idx2char[l] for l in list(ctc_output)])
                train_cer_result_list.append(cer_result)

        # 保存模型
        asr_train_model.save(f'{pb_path}/frame_{model_name}_{epoch_index}', save_format='h5')
        asr_train_model.save(f'{pb_path}/pb/frame.h5', save_format='tf')
        delete_frame(pb_path,model_name)

        # 验证集
        cer_result_list = []
        label_txt, pre_txt = '', ''
        for data in tqdm(test_loader):
            audio_features, audio_feature_lens, label_ids, label_lens = data

            asr_output = asr_train_model(audio_features, training=False)

            for i in range(len(asr_output)):
                asr_result = asr_output[i][:audio_feature_lens[i]]
                label_id = label_ids[i][:label_lens[i]]

                ctc_output = greedy_decode(asr_result, blank_index)[1]
                cer_result = edit(list(ctc_output), label_id) / len(label_id)

                label_txt = ''.join([idx2char[l] for l in label_id])
                pre_txt = ''.join([idx2char[l] for l in list(ctc_output)])
                cer_result_list.append(cer_result)

        print(np.mean(train_cer_result_list), np.mean(cer_result_list), label_txt, pre_txt)

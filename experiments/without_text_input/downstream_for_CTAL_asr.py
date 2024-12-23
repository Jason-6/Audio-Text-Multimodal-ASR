import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaConfig, RobertaModel
from m2p_model import AcousticModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import math

class RobertaM2Upstream(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')
        self.acoustic_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        # self.semantic_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])

        acoustic_model = AcousticModel(self.acoustic_config)
        semantic_model = RobertaModel(self.semantic_config, add_pooling_layer=False)

        # load the model from pretrained states
        self.acoustic_model = self.load_model(acoustic_model, ckpt_states['acoustic_model'], 'acoustic.')
        # self.semantic_model = self.load_model(semantic_model, ckpt_states['semantic_model'], 'roberta.')

    def load_model(self, transformer_model, state_dict, prefix_name=''):
        try:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            load(transformer_model, prefix_name)
            if len(missing_keys) > 0:
                print('Weights of {} not initialized from pretrained model: {}'.format(
                    transformer_model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                print('Weights from pretrained model not used in {}: {}'.format(
                    transformer_model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                    transformer_model.__class__.__name__, '\n\t'.join(error_msgs)))
            print('[CTAL] - {} Pre-trained weights loaded!'.format(prefix_name))
            return transformer_model

        except:
            raise RuntimeError('[CTAL] - {} Pre-trained weights NOT loaded!'.format(prefix_name))

    def forward(self,
                 # s_inputs=None,
                 # s_attention_mask=None,
                 # s_token_type_ids=None,
                 # s_position_ids=None,
                 # s_head_mask=None,
                 a_inputs=None,
                 a_attention_mask=None,
                 a_token_type_ids=None,
                 a_position_ids=None,
                 a_head_mask=None,
                 output_attentions=False,
                 output_hidden_states=False,
                 return_dict=None):

        # semantic_outputs = self.semantic_model(
        #     s_inputs,
        #     attention_mask=s_attention_mask,
        #     token_type_ids=s_token_type_ids,
        #     position_ids=s_position_ids,
        #     head_mask=s_head_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        # semantic_encode = semantic_outputs[0]

        acoustic_outputs = self.acoustic_model(
            a_inputs,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=a_position_ids,
            head_mask=a_head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        acoustic_encode = acoustic_outputs[0]

        return acoustic_encode



class ASR(nn.Module):
    ''' ASR model, including Encoder/Decoder(s)'''

    def __init__(self,ckpt_path,vocab_size, attention, decoder,emb_drop=0.0):
        super(ASR, self).__init__()

        # Setup
        self.vocab_size = vocab_size
        self.enable_att = 1
        self.lm = None

        # Modules
        """
        dec_dim: decoder's dim(512)
        encoder.out_dim:768，是config中的encoder的dim
        query_dim: 768，orignal = 512
        encode.out_dim:1024 = 512*2
        """
        self.encoder = RobertaM2Upstream(ckpt_path=ckpt_path).cuda()
        # print(self.encoder.state_dict())
        if self.enable_att == 1:
            self.dec_dim = decoder['dim']
            out_dim = 768
            self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
            """
            pre_embed: 输入的离散数据（例如文本中的标记）嵌入到一个低维向量空间中，这个向量空间的维度是由dec_dim指定的。
            这个Embedding层是在模型中用于学习和更新参数的一部分，
            用于将输入标记转化为模型内部的低维向量表示，以便进行后续计算。
            """
            self.embed_drop = nn.Dropout(emb_drop)
            self.decoder = Decoder(
                out_dim + self.dec_dim, vocab_size,**decoder)
            query_dim = self.dec_dim * self.decoder.layer
            self.attention = Attention(
                out_dim, query_dim, **attention)
        # if init_adadelta:
        #     self.apply(init_weights)
        #     if self.enable_att:
        #         for l in range(self.decoder.layer):
        #             bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
        #             bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        self.decoder.set_state(prev_state)
        self.attention.set_mem(prev_attn)

    def forward(self,text_inputs,
                    text_attention_mask,
                    speech_inputs,
                    speech_attention_mask,
                    real_len,
                    batch_size,
                    decode_step,
                    tf_rate = 0.0,
                    teacher=None,
                    emb_decoder=None,
                    get_dec_state=False):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxL] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
            real_len :其实就是原来的encode_len,因为编码器的输出序列长度是padding过的，需要获取到真实的，
                      因为数据经过模型不会被修改序列长度（目前text_model是不会），
                      所以直接在dataloader中计算出真实的长度，返回就可以了。
        '''
        # Init
        bs = batch_size  #
        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if get_dec_state else None

        # Encode
        # encode_feature, encode_len = self.encoder(audio_feature, feature_len)
        encoder_feature = self.encoder(
                a_inputs=speech_inputs,
                a_attention_mask=speech_attention_mask)


        # print("encoder-feature:::",encoder_feature)
        # Attention based decoding
        if self.enable_att:
            """
            如果启用了注意力机制，则初始化注意力解码器的状态（初始字符为<SOS>???），
            重置所有RNN状态和单元格，重置注意力机制的记忆，然后使用预处理数据进行teacher forcing。
            """
            # Init (init char = <SOS>, reset all rnn state and cell)
            self.decoder.init_state(bs)  # set to zero
            self.attention.reset_mem()
            last_char = self.pre_embed(torch.zeros(
                (bs), dtype=torch.long, device=encoder_feature.device))
            """
            last_char:
            创建一个大小为(bs, self.dec_dim)的张量last_char，用于表示每个样本的最后一个字符的嵌入向量。
            其中，所有元素都被初始化为0。这是因为在模型的初始状态下，我们并不知道输入序列的内容，因此无法对其进行嵌入操作。
            在模型的后续运行中，last_char将会被更新为实际的嵌入向量。
            最后，我们将last_char移动到与输入张量相同的计算设备上（如GPU），以确保在模型训练和推理期间可以高效地使用。
            """
            att_seq, output_seq = [], []

            # Preprocess data for teacher forcing
            if teacher is not None:
                teacher = self.embed_drop(self.pre_embed(teacher))

            # Decode
            for t in range(decode_step):
                # Attend (inputs current state of first layer, encoded features)
                attn, context = self.attention(
                    self.decoder.get_query(), encoder_feature, real_len)
                """
                对于目标序列中的每个时间步（t），首先执行一个注意力机制，通过将当前解码器的隐藏状态（self.decoder.get_query()）
                与编码器的输出（encode_feature）作为输入，得到当前时间步的注意力分布（attn）和上下文向量（context）。
                """
                # Decode (inputs context + embedded last character)
                decoder_input = torch.cat([last_char, context], dim=-1)
                cur_char, d_state = self.decoder(decoder_input)
                """
                将上下文向量与前一个时间步的输出字符的嵌入向量拼接成一个新的向量decoder_input，作为当前时间步的输入。
                具体来说，使用torch.cat函数将两个向量沿着最后一个维度（即特征维度）进行拼接，形成一个大小为(batch_size, dec_dim + enc_dim)的张量decoder_input。
                然后，将decoder_input作为输入，通过解码器（self.decoder）
                得到当前时间步的输出字符（cur_char）和新的解码器状态（d_state）。
                """
                # Prepare output as input of next step

                # Inference stage
                """
                否则进行自采样，即使用当前时间步的输出字符的分布进行采样得到一个字符，作为解码器的下一个输入字符。
                在每个时间步结束时，将当前时间步的输出字符的嵌入向量更新为last_char，以备下一个时间步使用。
                """
                # Prepare output as input of next step
                if (teacher is not None): ####给出目标序列即采用teacher-forcing的方式，可以不使用那就是自回归式的
                    # Training stage
                    if (tf_rate == 1) or (torch.rand(1).item() <= tf_rate):
                        # teacher forcing
                        last_char = teacher[:, t, :]
                    else:
                        # self-sampling (replace by argmax may be another choice)
                        with torch.no_grad():
                            if (emb_decoder is not None) and emb_decoder.apply_fuse:
                                _, cur_prob = emb_decoder(
                                    d_state, cur_char, return_loss=False)
                            else:
                                cur_prob = cur_char.softmax(dim=-1)
                            sampled_char = Categorical(cur_prob).sample()
                        last_char = self.embed_drop(
                            self.pre_embed(sampled_char))
                else:
                    # Inference stage
                    """
                    否则进行自采样，即使用当前时间步的输出字符的分布进行采样得到一个字符，作为解码器的下一个输入字符。
                    在每个时间步结束时，将当前时间步的输出字符的嵌入向量更新为last_char，以备下一个时间步使用。
                    """
                    if (emb_decoder is not None) and emb_decoder.apply_fuse:
                        _, cur_char = emb_decoder(
                            d_state, cur_char, return_loss=False)
                    # argmax for inference
                    last_char = self.pre_embed(torch.argmax(cur_char, dim=-1))#在自采样的情况下使用得到的采样字符的嵌入向量作为解码器的下一个输入
                # save output of each step
                output_seq.append(cur_char)
                att_seq.append(attn)
                if get_dec_state:
                    dec_state.append(d_state)

            att_output = torch.stack(output_seq, dim=1)  # BxTxV
            att_seq = torch.stack(att_seq, dim=2)  # BxNxDtxT
            """
            att_output: 是注意力机制的输出序列，其形状为 (batch_size, decode_step, vocab_size)。
            其中 batch_size 表示批次大小，decode_step 表示解码器的步骤数，vocab_size 表示词汇表大小。
            att_output 的每个元素代表当前时间步预测的下一个字符的概率分布。

            att_seq ：是注意力机制的上下文向量序列，其形状为 (batch_size, num_attn_layers, attn_dim, decode_step)。
            其中 num_attn_layers 表示注意力机制的层数，attn_dim 表示上下文向量的维度，decode_step 表示解码器的步骤数。
            att_seq 的每个元素表示注意力机制在当前时间步生成的上下文向量，用于解码器的下一个时间步。
            注意到 att_seq 的最后两个维度发生了交换，这是因为 stack 函数在调用时指定了 dim=2。
            """
            if get_dec_state:
                dec_state = torch.stack(dec_state, dim=1)

        return real_len, att_output, att_seq, dec_state


class Decoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''

    # ToDo:　More elegant way to implement decoder

    def __init__(self, input_dim, vocab_size, module, dim, layer, dropout):
        super(Decoder, self).__init__()
        self.in_dim = input_dim #在ASR类中传入了，input_dim = encode.out_dim + dec_dim,其中dec_dim是在config中的为512
        self.layer = layer #config:layer =1
        self.dim = dim  #config:dim=512
        self.dropout = dropout  #config:dropout=0

        # Init
        assert module in ['LSTM', 'GRU'], NotImplementedError
        self.hidden_state = None
        self.enable_cell = module == 'LSTM'

        # Modules
        self.layers = getattr(nn, module)(
            input_dim, dim, num_layers=layer, dropout=dropout, batch_first=True)
        """
        self.layers:
        建的神经网络层的名称可能是module，它的输入维度为input_dim，输出维度为dim，
        具有layer个层次结构。此外，该网络具有dropout概率，可用于防止过拟合，
        batch_first参数表示输入数据的第一维是否是批次维度。
        getattr是Python中的内置函数，用于获取对象（在此处为nn模块）的属性（在此处为module）。
        """
        self.char_trans = nn.Linear(dim, vocab_size)
        self.final_dropout = nn.Dropout(dropout)

    def init_state(self, bs):
        ''' Set all hidden states to zeros '''
        """
         如果Decoder使用的是LSTM，self.enable_cell被设置为True。
         如果使用的是LSTM模块（self.enable_cell为True），那么hidden_state是一个元组，包含两个张量，每个张量的形状为(layer, bs, dim)，
         表示LSTM模型的初始隐状态h0和记忆细胞c0。

        如果使用的是GRU模块（self.enable_cell为False），那么hidden_state是一个张量，形状为(layer, bs, dim)，表示GRU模型的初始隐状态。
        """
        device = next(self.parameters()).device
        """
        self.parameters() 方法返回了当前模型的所有参数列表，每个参数都是一个 PyTorch 张量（tensor）对象，
        其中包含了权重（weights）和偏置（bias）等模型参数。
        这些参数张量可能存储在 CPU 或 GPU 上
        next() 函数返回了参数列表中的第一个张量，并使用 .device 属性获取该张量的设备，
        然后将设备存储在 device 变量中。由于参数张量的设备可能不同，
        因此在遍历参数列表时，每个参数的设备可能会有所不同。
        """
        if self.enable_cell:
            self.hidden_state = (torch.zeros((self.layer, bs, self.dim), device=device),
                                 torch.zeros((self.layer, bs, self.dim), device=device))
        else:
            self.hidden_state = torch.zeros(
                (self.layer, bs, self.dim), device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        ''' Set all hidden states/cells, for decoding purpose'''
        """
        在set_state方法中，传入一个hidden_state参数，这个参数包含了一个元组或者一个张量，表示Decoder的隐状态。
        如果使用的是LSTM模块，hidden_state应该是一个元组，包含了两个张量，表示LSTM的隐状态和记忆细胞；
        如果使用的是GRU模块，hidden_state应该是一个张量，表示GRU的隐状态。

        set_state方法将输入的hidden_state复制到self.hidden_state中，以便在解码时使用。
        同时，为了将self.hidden_state放到正确的设备上，需要使用to方法将hidden_state转移到与模型参数相同的设备上。
        """
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (hidden_state[0].to(
                device), hidden_state[1].to(device))
        else:
            self.hidden_state = hidden_state.to(device)

    def get_state(self):
        ''' Return all hidden states/cells, for decoding purpose'''
        if self.enable_cell:
            return (self.hidden_state[0].cpu(), self.hidden_state[1].cpu())
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        ''' Return state of all layers as query for attention '''
        if self.enable_cell:
            return self.hidden_state[0].transpose(0, 1).reshape(-1, self.dim * self.layer)
        else:
            return self.hidden_state.transpose(0, 1).reshape(-1, self.dim * self.layer)

    def forward(self, x):
        ''' Decode and transform into vocab '''

        if not self.training:
            self.layers.flatten_parameters()  # 训练模式的时候，将rnn模型的参数“展平”，便于优化其反向传播计算
        """
        这行代码将输入张量x（在这里假设为RNN的隐藏状态）传递给RLSTM的layers层，同时传递先前的隐藏状态self.hidden_state。
        这将计算出新的隐藏状态，并将其作为新的self.hidden_state返回。
        在这里，我们将x增加了一个维度，以便它可以与LSTM的输入序列对齐。

        这行代码将x传递给字符级的全连接层self.char_trans。这将生成一个向量，其长度等于输出词汇表的大小。这个向量将用于预测下一个字符。
        在这里，还应用了一个dropout层，以避免过拟合。

        最后，函数返回了两个张量char和x，分别表示下一个字符的概率分布和最后一个隐藏状态的输出。
        """
        x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
        x = x.squeeze(1)  # 这行代码从x的第二个维度中删除大小为1的维度，因为我们不再需要它。
        char = self.char_trans(self.final_dropout(x))
        return char, x


class Attention(nn.Module):
    ''' Attention mechanism
        please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
        Input : Decoder state                      with shape [batch size, decoder hidden dimension]
                Compressed feature from Encoder    with shape [batch size, T, encoder feature dimension]
        Output: Attention score                    with shape [batch size, num head, T (attention score of each time step)]
                Context vector                     with shape [batch size, encoder feature dimension]
                (i.e. weighted (by attention score) sum of all timesteps T's feature) '''

    def __init__(self, v_dim, q_dim, mode, dim, num_head, temperature, v_proj,
                 loc_kernel_size, loc_kernel_num):
        super(Attention, self).__init__()

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.mode = mode.lower()
        self.num_head = num_head

        # Linear proj. before attention
        self.proj_q = nn.Linear(q_dim, dim * num_head)
        self.proj_k = nn.Linear(v_dim, dim * num_head)
        self.v_proj = v_proj
        if v_proj:
            self.proj_v = nn.Linear(v_dim, v_dim * num_head)

        # Attention
        if self.mode == 'dot':
            self.att_layer = ScaleDotAttention(temperature, self.num_head)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(
                loc_kernel_size, loc_kernel_num, dim, num_head, temperature)
        else:
            raise NotImplementedError

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(v_dim * num_head, v_dim)

        # Stored feature
        self.key = None
        self.value = None
        self.mask = None

    def reset_mem(self):
        self.key = None
        self.value = None
        self.mask = None
        self.att_layer.reset_mem()

    def set_mem(self, prev_attn):
        self.att_layer.set_mem(prev_attn)

    def forward(self, dec_state, enc_feat, enc_len):

        # Preprecessing
        bs, ts, _ = enc_feat.shape
        query = torch.tanh(self.proj_q(dec_state))
        query = query.view(bs, self.num_head, self.dim).view(
            bs * self.num_head, self.dim)  # BNxD

        if self.key is None:
            # Maskout attention score for padded states
            self.att_layer.compute_mask(enc_feat, enc_len.to(enc_feat.device))

            # Store enc state to lower computational cost
            self.key = torch.tanh(self.proj_k(enc_feat))
            self.value = torch.tanh(self.proj_v(
                enc_feat)) if self.v_proj else enc_feat  # BxTxN

            if self.num_head > 1:
                self.key = self.key.view(bs, ts, self.num_head, self.dim).permute(
                    0, 2, 1, 3)  # BxNxTxD
                self.key = self.key.contiguous().view(bs * self.num_head, ts, self.dim)  # BNxTxD
                if self.v_proj:
                    self.value = self.value.view(
                        bs, ts, self.num_head, self.v_dim).permute(0, 2, 1, 3)  # BxNxTxD
                    self.value = self.value.contiguous().view(
                        bs * self.num_head, ts, self.v_dim)  # BNxTxD
                else:
                    self.value = self.value.repeat(self.num_head, 1, 1)

        # Calculate attention
        context, attn = self.att_layer(query, self.key, self.value)
        if self.num_head > 1:
            context = context.view(
                bs, self.num_head * self.v_dim)  # BNxD  -> BxND
            context = self.merge_head(context)  # BxD

        return attn, context


class BaseAttention(nn.Module):
    ''' Base module for attentions '''

    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self, prev_att):
        pass

    def compute_mask(self, k, k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs, ts, _ = k.shape
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[idx, :, sl:] = 1  # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(
            k_len.device, dtype=torch.bool).view(-1, ts)  # BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn)  # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(
            1)  # BNxT x BNxTxD-> BNxD
        return output, attn


class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(
            1, 2)).squeeze(1)  # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy, v)

        attn = attn.view(-1, self.num_head, ts)  # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''
    """
    该代码实现了一个基于位置的注意力机制（Location-Aware Attention），其基本思想是将先前时刻的注意力分布作为位置信息，
    与当前时刻的输入共同影响注意力分布的计算。该模块的输入为查询向量q、键向量k和值向量v，输出为加权后的值向量和注意力分布。

    具体来说，该模块首先通过一维卷积（nn.Conv1d）将先前时刻的注意力分布转换为一个新的位置向量，然后通过线性变换（nn.Linear）
    将其映射到与输入向量相同的维度。接着，将查询向量、键向量、位置向量的和作为输入，计算注意力分布。最后，将注意力分布与值向量进行加权，得到加权和向量作为输出。

    该模块还提供了两个方法reset_mem和set_mem，用于重置和设置先前时刻的注意力分布

    """

    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att = None
        self.loc_conv = nn.Conv1d(
            num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim, bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh, ts, _ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs, self.num_head, ts)).to(k.device)
            for idx, sl in enumerate(self.k_len):
                self.prev_att[idx, :, :sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(
            self.prev_att).transpose(1, 2)))  # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(
            1, self.num_head, 1, 1).view(-1, ts, self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1)  # BNx1xD

        # Compute energy and context
        energy = self.gen_energy(torch.tanh(
            k+q+loc_context)).squeeze(2)  # BNxTxD -> BNxT
        output, attn = self._attend(energy, v)
        attn = attn.view(bs, self.num_head, ts)  # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn

def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3, 4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError


def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    return bias

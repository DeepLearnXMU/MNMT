from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
               initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        #默认是单向的
        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            decoder_hidden_size = hidden_size
            self.bridge_nn = nn.Linear(1, decoder_hidden_size)
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
            # print(encoder_final.size())
            # encoder_final=self.mean_bridge(encoder_final)
            # print(encoder_final.size())
        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for i in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def mean_bridge(self, bridge_input):
        '''
        bridge_input:[num_direction x batch x standar_hidden_size/num_direction] 如[2, 32 ,128]
        '''
        # bridge_input = torch.cat((bridge_input[0],bridge_input[1]),1).squeeze(0) #[batch, hidden_size]
        mean_outputs = torch.mean(bridge_input,2).unsqueeze(2)  #[num_direction, batch, 1]
        
        return torch.tanh(self.bridge_nn(mean_outputs.view(-1,1)))\
                    .view(bridge_input.size())

    
        
class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.



    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False,gpu=None,image_feat_type=None,bi_dim_attn=1,co_attention=1):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(0.7)
        # self.bi_dim_attn = bi_dim_attn
        self.dropout_memory_bank = nn.Dropout(0.05)
        self.dropout_hidden=nn.Dropout(dropout)
        self.bi_attention=bi_dim_attn
        self.co_attention = co_attention

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)
        rec2_input_size=hidden_size if self.co_attention==1 else hidden_size
        self.rec2 = self._build_rnn(rnn_type,
                                   input_size=rec2_input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)
        # self.Linear_gate_theta=nn.Linear(hidden_size,1)
        self.sigmoid=nn.Sigmoid()
        self.sm = nn.Softmax()
        # self.Linear_bi_dim_attn = nn.Linear(self.hidden_size, 1)
        # self.gate_bi_dim_attn_memory = nn.Linear(hidden_size,1)
        # self.gate_context_img = nn.Linear(hidden_size ,1)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )
        # self.context_gate1=onmt.modules.context_gate_factory(
        #         'both', self._input_size,
        #         hidden_size, hidden_size, hidden_size
        #     )
        # self.context_gate2=onmt.modules.context_gate_factory(
        #         'both', hidden_size,
        #         hidden_size, hidden_size, hidden_size
        #     )
        # Set up the standard attention.
        self._coverage = coverage_attn

        self.attn_no_guiding=onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type,gpu=gpu
        )

        self.attn_img = onmt.modules.GlobalAttention(
            hidden_size,
            coverage=coverage_attn, # coverage not yet implemented for visual attention
            attn_type=attn_type,gpu=gpu, multi_query=True
        )

        if self.co_attention == 1:
            self.Linear_gate_beta=nn.Linear(hidden_size,1)
            self.attn = onmt.modules.GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type,gpu=gpu, multi_query=True
            )


        if self.bi_attention == 1:
            self.Linear_visual2text=nn.Linear(hidden_size, 1)
            self.Linear_text2visual=nn.Linear(hidden_size, 1)
            #图像文本引导图像
            self.bi_attn_text_guiding_img =  onmt.modules.GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type='dot',gpu=gpu, multi_query=False
            )

            # 图像文本引导文本
            self.bi_attn_text_guiding_img2 = onmt.modules.GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type='dot', gpu=gpu, multi_query=False
            )

        

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None, context_img=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
            context_img (`FloatTensor`): vectors from the image
                 `[batch * img_feats_len x hidden]`
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        src_len = memory_bank.size(0)
        memory_bank = memory_bank.transpose(0, 1).contiguous()  # [batch, src_len, hidden]

        if context_img is not None and self.bi_attention:
            ###############bi-dimension attention############
            memory_bank = self.dropout_hidden(memory_bank)
            img_feat_num = context_img.size(1)
            img_text_alignment= self.bi_attn_text_guiding_img( context_img.contiguous(),memory_bank,memory_lengths)       #[batch, img_feat_num, src_len]  还没有softmax完的

            text2visual_alignment = self.sm(img_text_alignment.transpose(1,2).contiguous().view(tgt_batch*src_len, img_feat_num))\
                                        .view(tgt_batch, src_len, img_feat_num) #[batch, src_len, img_feat_num]
            text2visual = torch.bmm(text2visual_alignment, context_img) #[batch, src_len, hidden]

            img_text_alignment2 = self.bi_attn_text_guiding_img2(context_img.contiguous(), memory_bank,
                                                               memory_lengths)  # [batch, img_feat_num, src_len]  还没有softmax完的

            if memory_lengths is not None:
                mask = sequence_mask(memory_lengths)
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                img_text_alignment2.data.masked_fill_(1 - mask, -float('inf'))

            visual2text_alignment = self.sm(img_text_alignment2.view(tgt_batch*img_feat_num, src_len))\
                            .view(tgt_batch, img_feat_num, src_len)
            visual2text = torch.bmm(visual2text_alignment,memory_bank)  #[batch, img_feat, hidden]

            #整合
            gate_text2visual= self.Linear_text2visual(memory_bank)#self.Linear_text2visual(torch.cat((memory_bank ,text2visual),2))
            memory_bank = memory_bank+gate_text2visual*text2visual

            gate_visual2text =self.Linear_visual2text(context_img)#self.Linear_visual2text(torch.cat((context_img ,visual2text),2 ))
            context_img = context_img + gate_visual2text * visual2text



        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final,context_img=None):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder :
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final]),context_img)
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final),context_img)


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        decoder_outputs, p_attn = self.attn_no_guiding(
            rnn_output.transpose(0,1).contiguous(),
            memory_bank.contiguous(),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None,context_img=None):
        """
        context_img:[batch x len x feats]
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # memory_bank = self.dropout_memory_bank(memory_bank)
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
        if context_img is not None:
            attns["img"]=[]
            attns["std2"]=[]
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        # print(len(hidden))
        # print(hidden[0].size())
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None
        
        # memory_bank = memory_bank.transpose(0,1)
        # memory_bank=self.dropout_hidden(memory_bank)
      
        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input=emb_t
            

            #rnn_output：最后一层的所有
            #hidden：    所有层的最后一个
            rnn_output, hidden = self.rnn(decoder_input, hidden)
           
           
            #用decoder的隐层作为guiding
            #decoder_output其实就是context  
            text_context, p_attn = self.attn_no_guiding(
                rnn_output,
                memory_bank,
                memory_lengths=memory_lengths)
            # text_context = self.dropout_hidden(text_context)
            # print(p_attn)
            # raise AssertionError
            
            # attn_output_img=None
            if self.co_attention == 0:
                img_context, attn_img = self.attn_img(
                    rnn_output,
                    context_img,
                    memory_lengths=None,  # 不进行mask
                )
                if context_img is None:
                    rec2_input = text_context + img_context
                    decoder_output, hidden = self.rec2(rec2_input, hidden)
                else:
                    decoder_output = text_context
             

                if self.context_gate is not None:
                    decoder_output = self.context_gate(
                            decoder_input, rnn_output, decoder_output
                        )
            else:
                if context_img is not None:
                    img_context, attn_img = self.attn_img(
                        rnn_output,
                        context_img,
                        memory_lengths=None,    #不进行mask
                        query_modal=text_context
                        )

                    gate_beta = self.sigmoid(self.Linear_gate_beta(rnn_output))
                    attn_output_img=torch.mul(img_context, gate_beta)

                    #&&&  这里可以考虑评上一些东西如：目标端的隐层
                    #用visual feature经过attention加权求和后的向量  、  decoder的隐层作为guiding
                    attn_output_1, attn = self.attn(
                        rnn_output,
                        memory_bank,
                        memory_lengths=memory_lengths,
                        coverage=coverage,
                        query_modal=attn_output_img)

                    # #利用context 来更新decoder的隐层
                    rec2_input = text_context + attn_output_img + attn_output_1
                    decoder_output, hidden = self.rec2(rec2_input, hidden)

                else:
                    decoder_output, hidden = self.rec2(text_context, hidden)

                if self.context_gate is not None:
                    decoder_output = self.context_gate(
                        decoder_input, text_context, decoder_output
                    )

            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            
            attns["std"] += [p_attn]
            if context_img is not None:
                attns['img']+=[attn_img]
                if self.co_attention==1:
                    attns['std2']+=[attn]
                else:
                    attns['std2'] += [p_attn]
            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size #+ self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder,encoder_img=None, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_img=encoder_img

    def forward(self, src, tgt, lengths, dec_state=None, context_img=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        # print('@@@Models ',type(context_img))
        tgt = tgt[:-1]  # exclude last target from inputs
        # project/transform local image features into the expected structure/shape
        img_proj=None
        if self.encoder_img is not None:
            # print('@@@Models context_img',context_img.size())
            img_proj = self.encoder_img( context_img )
            # print('@@@Models img_proj',img_proj.size())

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths,
                         context_img=img_proj)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate,context_img=None):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)
        # self.input_feed_img = Variable(context_img.data.new(*h_size).zero_(),
        #                                requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class ImageLocalFeaturesProjector(nn.Module):
    """
        Reshape local image features.
    """
    def __init__(self, num_layers, nfeats, outdim, dropout,
            use_nonlinear_projection):
        """
        Args:
            num_layers (int): 1.
            nfeats (int): size of image features.
            outdim (int): size of the output dimension.
            dropout (float): dropout probablity.
            use_nonliner_projection (bool): use non-linear activation
                    when projecting the image features or not.
        """
        super(ImageLocalFeaturesProjector, self).__init__()
        assert(num_layers==1), \
                'num_layers must be equal to 1 !'
        self.num_layers = num_layers
        self.nfeats = nfeats
        self.dropout = dropout
        
        layers = []
        # reshape input
        layers.append( View(-1, 14*14, nfeats) )
        # linear projection from feats to rnn size
        layers.append( nn.Linear(nfeats, outdim*num_layers) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        #self.batch_norm = nn.BatchNorm2d(512)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        #print "out.size(): ", out.size()
        #if self.num_layers>1:
        #    out = out.unsqueeze(0)
        #    out = torch.cat([out[:,:,0:out.size(2):2], out[:,:,1:out.size(2):2]], 0)
        #    #print "out.size(): ", out.size()
        return out


class ImageGlobalFeaturesProjector(nn.Module):
    """
        Project global image features using a 2-layer multi-layer perceptron.
    """
    def __init__(self, num_layers, nfeats, outdim, dropout,
            use_nonlinear_projection):
        """
        Args:
            num_layers (int): number of decoder layers.
            nfeats (int): size of image features.
            outdim (int): size of the output dimension.
            dropout (float): dropout probablity.
            use_nonliner_projection (bool): use non-linear activation
                    when projecting the image features or not.
        """
        super(ImageGlobalFeaturesProjector, self).__init__()
        self.num_layers = num_layers
        self.nfeats = nfeats
        self.outdim = outdim
        self.dropout = dropout
        
        layers = []
        layers.append( nn.Linear(nfeats, nfeats) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        # final layers projects from nfeats to decoder rnn hidden state size
        layers.append( nn.Linear(nfeats, outdim*num_layers) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        #self.batch_norm = nn.BatchNorm2d(512)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        #print "out.size(): ", out.size()
        if self.num_layers>1:
            out = out.unsqueeze(0)
            out = torch.cat([out[:,:,0:out.size(2):2], out[:,:,1:out.size(2):2]], 0)
            #print "out.size(): ", out.size()
        return out

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))



class View(nn.Module):
    """Helper class to be used inside Sequential object to reshape Variables"""
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

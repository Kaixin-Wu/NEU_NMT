import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from beam_search import Beam
from utils import get_threshold

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        stdv = get_threshold(input_size, embed_size)
        self.embed.weight.data.uniform_(-stdv, stdv)

        # self.gru = nn.GRU(embed_size, hidden_size, n_layers,
        #                  dropout=dropout, bidirectional=True)
        self.gru_forward = nn.GRU(embed_size, hidden_size, n_layers,
                                  dropout=dropout)
        self.gru_backward = nn.GRU(embed_size, hidden_size, n_layers,
                                   dropout=dropout)        
    
    def forward(self, src, len_src, hidden=None):
        '''
        :param src: [timestep ,batch]
        :param len_src:  [batch]
        :param hidden:
        :return:
        '''
        embedded = self.embed(src)
        outputs_forward, hidden_forward = self.gru_forward(embedded, hidden)

        max_len, batch = src.size()
        src_reversed = Variable(torch.LongTensor(batch, max_len)).cuda()  # [batch, timestep]
        
        src_trans = src.transpose(0, 1)  # [batch, timestep]
        for i in range(len_src.size(0)):
            end = int(len_src[i])

            idx = [j for j in range(max_len)]
            idx = idx[:end][::-1] + idx[end:]

            idx = torch.LongTensor(idx).cuda()
            src_reversed[i] = src_trans[i][idx]
       
        src_reversed = src_reversed.transpose(0, 1) # [timestep, batch]
        embedded_reversed = self.embed(src_reversed)
        outputs_backward, hidden_backward = self.gru_backward(embedded_reversed, hidden)

        outputs = torch.cat([outputs_forward, outputs_backward], 2)

        hidden = Variable(torch.zeros(batch, self.hidden_size)).cuda()
        tmp = outputs_backward.transpose(0, 1) # [batch, max_len, hidden_size]
        for i in range(tmp.size(0)):
            tmp_matrix = tmp[i]
            index = len_src[i]-1
            hidden[i] = tmp_matrix[index]

        hidden = hidden.unsqueeze(0) # [1, batch, hidden_size]

        return outputs, hidden
    # def forward(self, src, hidden=None):
    #     embedded = self.embed(src)
    #     self.gru.flatten_parameters()   ## Edit by Wu Kaixin 2018/1/9
    #     outputs, hidden = self.gru(embedded, hidden)

    #     '''
    #     # sum bidirectional outputs
    #     outputs = (outputs[:, :, :self.hidden_size] +
    #                outputs[:, :, self.hidden_size:])
    #     '''
    #     return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        stdv = get_threshold(hidden_size * 3, hidden_size)
        self.attn.weight.data.uniform_(-stdv, stdv)
        self.attn.bias.data.zero_()

        self.v = nn.Parameter(torch.rand(hidden_size))
        # stdv = 1. / math.sqrt(self.v.size(0))
        stdv = get_threshold(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs, len_src):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*2H]
        attn_energies = self.score(h, encoder_outputs)

        mask = Variable(torch.zeros(attn_energies.size(0), attn_energies.size(1))).cuda()
        for i in range(len_src.size(0)):
            index = len_src[i]
            if int(index) < mask.size(1):
                mask[i, int(index):] = float("-inf")
        attn_energies += mask        

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
        energy = F.tanh(energy.transpose(1, 2))  # [B*H*T] # add F.tanh  2018/1/11
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        stdv = get_threshold(output_size, embed_size)
        self.embed.weight.data.uniform_(-stdv, stdv)

        ### self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 3, output_size)
        stdv = get_threshold(hidden_size * 3, output_size)
        self.out.weight.data.uniform_(-stdv, stdv)
        self.out.bias.data.zero_()

    def forward(self, input, last_hidden, encoder_outputs, len_src):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        ### embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs, len_src)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        
        self.gru.flatten_parameters()  ## Edit by Wu Kaixin 2018/1/9
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        '''
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights
        '''
        output = torch.cat([output, context], 1)  ## [batch, 3hidden_size]  Edit by Wu kaixin  2018/1/10
        output = F.tanh(output)  ## Wu Kaixin 2018/1/11
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, len_src):
        '''
        :param src:  [src_max_len, batch]
        :param trg:  [trg_max_len, batch]
        :param len_src: [batch]
        :return:
        '''
        encoder_output, hidden = self.encoder(src, len_src)
        '''
            ## src: [src_max_len, batch]
            ## encoder_output: [src_max_len, batch, hidden_size]
            ## hidden: (num_layers * num_directions, batch, hidden_size) -> [2, batch, hidden_size]
        '''
        hidden = hidden[:self.decoder.n_layers]

        batch_size = src.size(1)
        max_len = trg.size(0)
        hidden_size = self.decoder.hidden_size
        ## vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len-1, batch_size, 3 * hidden_size)).cuda()

        output = Variable(trg.data[0, :]) # sos [batch]
        for t in range(1, max_len):
            # output: [batch] -> [batch, 3hidden_size]
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output, len_src)

            outputs[t-1] = output
            output =Variable(trg.data[t]).cuda()

        transform_output = self.decoder.out(outputs) # [max_len-1, batch, output_size]
        softmax_output = F.log_softmax(transform_output, dim=2)  # [max_len-1, batch, output_size]

        return softmax_output
    
    '''
    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len-1, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            self.decoder.flatten_parameters()  ## Edit by Wu Kaixin 2018/1/9
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t-1] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs # [max_len-1, batch, vocab_size]
    '''

    def translate(self, src, len_src, trg, beam_size, Lang2):
        ''' beam search decoding. '''
        '''
        :param src:   [src_max_len, batch]    ## batch = 1
        :param trg:   [trg_max_len, batch]    ## batch = 1
        :param sentence:  [sentence_len]
        :return: best translate candidate
        '''
        max_len = trg.size(0)
        encoder_output, hidden = self.encoder(src, len_src)
        '''
            ## src: [src_max_len, batch]
            ## encoder_output: [src_max_len, batch, hidden_size]
            ## hidden: (num_layers * num_directions, batch, hidden_size) -> [2, batch, hidden_size]
        '''
        hidden = hidden[:self.decoder.n_layers]  # [n_layers, batch, hidden_size]
        # trg: [trg_max_len, batch]
        output = Variable(trg.data[0, :])  # sos  [batch]

        beam = Beam(beam_size, Lang2.vocab.stoi, True)
        for t in range(1, max_len):
            # output:  [batch] -> [batch, output_size]
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output, len_src)
            output = self.decoder.out(output)
            output = F.log_softmax(output, dim=1)

            workd_lk = output
            if output.size(0) == 1:

                output_prob = output.squeeze(0) ## [output_size]
                workd_lk = output_prob.expand(beam_size, output_prob.size(0))  ## [beam_size, output_size]

                # [n_layers, batch, hidden_size]
                hidden = hidden.squeeze(1)  # [n_layers, hidden_size]
                hidden = hidden.expand(beam_size, hidden.size(0), hidden.size(1)) # [beam_size, n_layers, hidden_size]
                hidden = hidden.transpose(0, 1) # [n_layers, beam_size, hidden_size]
                
                # [src_max_len, batch, hidden_size]
                encoder_output = encoder_output.squeeze(1) ## [src_max_len, hidden_size]
                encoder_output = encoder_output.expand(beam_size, encoder_output.size(0), encoder_output.size(1)) ## [beam_size, src_max_len, hidden_size]
                encoder_output = encoder_output.transpose(0, 1)  ## [src_max_len, beam_size, hidden_size] 

            flag = beam.advance(workd_lk)
            if flag:
                break

            nextInputs = beam.get_current_state()
            # print("[nextInputs]:", nextInputs)
            output = nextInputs
            # output = Variable(nextInputs).cuda()

            originState = beam.get_current_origin()
            ## print("[origin_state]:", originState)
            hidden = hidden[:, originState]

        xx, yy = beam.get_best()
        zz = beam.get_final()
        return xx, yy, zz

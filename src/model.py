import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from beam_search import Beam

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)

        '''
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        '''
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*2H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
        energy = energy.transpose(1, 2)  # [B*H*T]
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
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 3, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs

    def translate(self, src, trg, beam_size, Lang2):
        ''' beam search decoding. '''
        '''
        :param src:   [src_max_len, batch]    ## batch = 1
        :param trg:   [trg_max_len, batch]    ## batch = 1
        :param sentence:  [sentence_len]
        :return: best translate candidate
        '''
        max_len = trg.size(0)
        encoder_output, hidden = self.encoder(src)
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
                    output, hidden, encoder_output)

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
            print("[origin_state]:", originState)
            hidden = hidden[:, originState]

        xx, yy = beam.get_best()
        zz = beam.get_final()
        return xx, yy, zz

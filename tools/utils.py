import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim 
from torch.autograd import Variable

#Avoid seeing future words
#Prediction at position i, can only depend on inputs less than i
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0


#Restricted window
#Also works in the case where the query Seq != key seq
def relative_window(size, window=20):

    rel_mask = torch.zeros((size, size))

    for i in range(size):
        left = i-(window) if i >= (window) else 0
        right = i+(window) if i <= size - (window) else size
        rel_mask[i, left:right] = 1

    return rel_mask


#Use this to create data masks
#src_mask : (batch, 1, seq)
#tgt_mask : (batch, seq, seq)
#rel_mask : (batch, seq, seq)
class Batch:

    def __init__(self, src_lengths, hand_lengths, trg=None, eos_index=2, pad=0, DEVICE='cuda', emb_type='2d', fixed_padding=None, rel_window=None):

        #No need for mask in 3d embeddings
        if(emb_type == '3d'):
            self.src_mask = None

        elif(emb_type == '2d'):
            #NOTE: Create a mask for src sequence to hide padding
            if(fixed_padding):
                self.src_mask = torch.zeros((len(src_lengths), fixed_padding), dtype=torch.uint8).to(DEVICE)
            else:
                self.src_mask = torch.zeros((len(src_lengths), np.amax(src_lengths)), dtype=torch.uint8).to(DEVICE)

            for i, length in enumerate(src_lengths):
                self.src_mask[i, :length] = 1

            self.src_mask = self.src_mask.unsqueeze(-2)

            #Local window masking
            if(rel_window):
                #rel_mask = relative_window(np.amax(src_lengths), np.amax(hand_lengths), rel_window)
                rel_mask = relative_window(np.amax(src_lengths), rel_window)

                #Padding input images
                self.rel_mask = self.src_mask * Variable(rel_mask.type_as(self.src_mask.data))
            else:
                self.rel_mask = None

        else:
            print("embeddings not recognized!")

        if trg is not None:
            #Add <sos> token for input of decoder
            #NOTE:We shift the decoder input by one position, setting <sos> in the first position
            #BY doing this the decoder predicts next word instead of predicting the same word
            #in each time step. So we prevent the model from learning the copy/paste task
            #NOTE: Input of decoder (self.trg) (with sos and without eos)
            #ground truth target (self.y) (with eos and without sos)

            #Remove eos token from input of decoder

            self.trg = trg[:, :-1]

            for i in range(len(trg)):
                if(trg[i, -1] == pad):
                    self.trg[i, :] = trg[i, trg[i] != eos_index]

            #Create a matrix mask to avoid seeing future words in tgt
            self.trg_mask = self.make_std_mask(self.trg, pad)

            #Sum number of tokens that are not pad
            self.ntokens = (self.trg != pad).data.sum()

        #maximum seq_length of batch
        self.seq = np.amax(src_lengths)

    @staticmethod
    def make_std_mask(trg, pad):
        #NOTE: Create a mask for tgt sequence to hide padding and seeing future words.
        trg_mask = (trg != pad).unsqueeze(-2)
        trg_mask = trg_mask & Variable(
            subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
        #Make it 0s and 1s
        trg_mask = trg_mask.type(torch.uint8)
        return trg_mask


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
       
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        print("RATE")
        print(rate)
        self.optimizer.step()
        
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    #return NoamOpt(model.src_embed[0].d_model, 2, 4000,
     return NoamOpt(1280, 2, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


#http://nlp.seas.harvard.edu/2018/04/03/attention.html#label-smoothing
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=2891, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        #Size of vocab
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        loss = self.criterion(x, Variable(true_dist, requires_grad=False))
        del true_dist

        return loss

import argparse
import pickle as pkl
import util
import os
import time
import numpy as np

from Layers import EncoderLayer, DecoderLayer
from Embd import Embedder, PositionalEncoder
from Sublayers import Norm

import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import random



def nopeak_mask(trg_size, size, opt):
    np_mask = np.tril(np.ones((trg_size, size, size)),
    k=0).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask))
    if opt.gpuid>-1:
      np_mask = np_mask.cuda()
    return np_mask



def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


def eval_training(opt, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, using_gpu, form_manager):
    # encode, decode, backward, return loss

    batch_num = train_loader.get_batchnum()
    total_loss = 0
    for i in range(0, batch_num):

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        enc_batch, enc_len_batch, dec_batch = train_loader.ordered_batch(i)
        #print(enc_batch, enc_len_batch, dec_batch)
        #print(enc_batch.size(), len(enc_len_batch), dec_batch.size())
        # do not predict after <E>
        enc_max_len = enc_batch.size(1)

        #print(enc_max_len)
        enc_batch_mask = torch.ones(enc_batch.size(0), enc_max_len)
        for i in range(enc_batch_mask.size(0)):
            if enc_len_batch[i] < enc_max_len:
                enc_batch_mask[i, enc_len_batch[i]:enc_max_len] = 0
        # because you need to compare with the next token!!
        dec_max_len = dec_batch.size(1) - 1
        pred_function = nn.Linear(opt.rnn_size, form_manager.vocab_size)
        softmax_function = nn.LogSoftmax(dim=1)
        trg_input = dec_batch[:, :-1]
        #print(trg_input.size())
        src_mask = enc_batch_mask.unsqueeze(-2)
        trg_mask = nopeak_mask(opt.batch_size, dec_max_len, opt)
        #print(src_mask.size(), trg_mask.size())
        e_outputs = encoder(enc_batch, src_mask)
        d_output = decoder(trg_input, e_outputs, src_mask, trg_mask)
        preds = pred_function(d_output)
        #print(preds.size())
        ys = dec_batch[:, 1:].contiguous().view(-1)
        loss = criterion(preds.view(-1, preds.size(-1)), ys)
        loss = loss / opt.batch_size
        loss.backward()

        torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_value_(decoder.parameters(), opt.grad_clip)
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()
    return total_loss

def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)


def do_generate(encoder, decoder, enc_w_list, word_manager, form_manager, opt, using_gpu):
    # initialize the rnn state to all zeros
    enc_w_list.append(word_manager.get_symbol_idx('<S>'))
    enc_w_list.insert(0, word_manager.get_symbol_idx('<E>'))
    end = len(enc_w_list)
    prev_c  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    prev_h  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    enc_outputs = torch.zeros((1, end, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()
        enc_outputs = enc_outputs.cuda()
    # TODO check that c,h are zero on each iteration
    # reversed order
    for i in range(end-1, -1, -1):
        # TODO verify that this matches the copy_table etc in sample.lua
        cur_input = torch.tensor(np.array(enc_w_list[i]), dtype=torch.long)
        if using_gpu:
            cur_input = cur_input.cuda()
        prev_c, prev_h = encoder(cur_input, prev_c, prev_h)
        enc_outputs[:, i, :] = prev_h
    #encoder_outputs = torch.stack(encoder_outputs).view(-1, end, encoder.hidden_size)
    # decode
    if opt.sample == 0 or opt.sample == 1:
        text_gen = []
        if opt.gpuid >= 0:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long).cuda()
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long)
        while True:
            prev_c, prev_h = decoder(prev_word, prev_c, prev_h)
            pred = attention_decoder(enc_outputs, prev_h)
            #print("prediction: {}\n".format(pred))
            # log probabilities from the previous timestamp
            if opt.sample == 0:
                # use argmax
                _, _prev_word = pred.max(1)
                prev_word = _prev_word.resize(1)
            if (prev_word[0] == form_manager.get_symbol_idx('<E>')) or (len(text_gen) >= opt.dec_seq_length):
                break
            else:
                text_gen.append(prev_word[0])
        return text_gen

def check_validation(opt, encoder, decoder, using_gpu):
    encoder.eval()
    decoder.eval()
    # initialize the vocabulary manager to display text
    managers = pkl.load( open("{}/map.pkl".format(args.data_dir), "rb" ) )
    word_manager, form_manager = managers
    # load data
    data = pkl.load(open("{}/dev.pkl".format(args.data_dir), "rb"))
    reference_list = []
    candidate_list = []
    for i in range(len(data)):
        x = data[i]
        reference = x[1]
        candidate = do_generate(encoder, decoder, x[0], word_manager, form_manager, args, using_gpu)
        candidate = [int(c) for c in candidate]
        num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== "(")
        num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== ")")
        diff = num_left_paren - num_right_paren
        #print(diff)
        if diff > 0:
            for i in range(diff):
                candidate.append(form_manager.symbol2idx[")"])
        elif diff < 0:
            candidate = candidate[:diff]

        ref_str = convert_to_string(reference, form_manager)
        cand_str = convert_to_string(candidate, form_manager)

        reference_list.append(reference)
        candidate_list.append(candidate)
        # print to console

    val_acc = util.compute_tree_accuracy(candidate_list, reference_list, form_manager)
    print("ACCURACY = {}\n".format(val_acc))

def main(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    managers = pkl.load( open("{}/map.pkl".format(opt.data_dir), "rb" ) )
    word_manager, form_manager = managers
    using_gpu = False
    if opt.gpuid > -1:
        using_gpu = True
        torch.manual_seed(opt.seed)

    encoder = Encoder(opt, word_manager.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    decoder = Decoder(opt, form_manager.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    if using_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    # init parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)

    ##-- load data
    train_loader = util.MinibatchLoader(opt, 'train', using_gpu)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    ##-- start training
    step = 0
    epoch = 0
    optim_state = {"learningRate" : opt.learning_rate, "alpha":  opt.decay_rate}
    # default to rmsprop
    if opt.opt_method == 0:
        encoder_optimizer = optim.Adam(encoder.parameters(),  lr=optim_state["learningRate"])
        decoder_optimizer = optim.Adam(decoder.parameters(),  lr=optim_state["learningRate"])
    criterion = nn.CrossEntropyLoss(reduction='sum')

    print("Starting training.")
    encoder.train()
    decoder.train()
    start_time = time.time()
    iterations = opt.max_epochs
    for i in range(iterations):
        epoch = i // train_loader.num_batch
        train_loss = eval_training(opt, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, using_gpu, form_manager)

        if i % opt.print_every == 0:
            # if i > 0 and train_loss < 3:
            #     check_validation(opt, encoder, decoder, using_gpu)
            end_time = time.time()
            print("{}/{}, train_loss = {}, time/batch = {}".format(i, iterations, train_loss, (end_time - start_time)/60))
            start_time = time.time()

        #on last iteration
        if i == iterations -1:
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            checkpoint["opt"] = opt
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(checkpoint, "{}/model_transformer".format(opt.checkpoint_dir))

        if train_loss != train_loss:
            print('loss is NaN.  This usually indicates a bug.')
            break

if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=-1, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/', help='data path')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')
    main_arg_parser.add_argument('-checkpoint_dir',type=str, default= 'checkpoint_dir', help='output directory where checkpoints get written')
    main_arg_parser.add_argument('-savefile',type=str, default='save',help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
    main_arg_parser.add_argument('-print_every',type=int, default=1,help='how many steps/minibatches between printing out the loss')
    main_arg_parser.add_argument('-rnn_size', type=int,default=256, help='size of LSTM internal state')
    main_arg_parser.add_argument('-num_layers', type=int, default=1, help='number of layers in the LSTM')
    #main_arg_parser.add_argument('-dropout',type=float, default=0.4,help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    main_arg_parser.add_argument('-dropoutrec',type=int,default=0,help='dropout for regularization, used after each c_i. 0 = no dropout')
    main_arg_parser.add_argument('-enc_seq_length',type=int, default=200,help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-dec_seq_length',type=int, default=100,help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-batch_size',type=int, default=32,help='number of sequences to train on in parallel')
    main_arg_parser.add_argument('-max_epochs',type=int, default=480,help='number of full passes through the training data')
    main_arg_parser.add_argument('-opt_method', type=int,default=0,help='optimization method: 0-rmsprop 1-sgd')
    main_arg_parser.add_argument('-learning_rate',type=float, default=4e-4,help='learning rate')
    main_arg_parser.add_argument('-init_weight',type=float, default=0.08,help='initailization weight')
    main_arg_parser.add_argument('-learning_rate_decay',type=float, default=0.98,help='learning rate decay')
    main_arg_parser.add_argument('-learning_rate_decay_after',type=int, default=5,help='in number of epochs, when to start decaying the learning rate')
    main_arg_parser.add_argument('-restart',type=int, default=-1,help='in number of epochs, when to restart the optimization')
    main_arg_parser.add_argument('-decay_rate',type=float, default=0.95,help='decay rate for rmsprop')
    main_arg_parser.add_argument('-grad_clip',type=int, default=5,help='clip gradients at this value')
    main_arg_parser.add_argument('-sample', type=int, default=0, help='0 to use max at each timestep (-beam_size=1), 1 to sample at each timestep, 2 to beam search')
    main_arg_parser.add_argument('-beam_size', type=int, default=20, help='beam size')
    main_arg_parser.add_argument('-d_model', type=int, default=256)
    main_arg_parser.add_argument('-n_layers', type=int, default=6)
    main_arg_parser.add_argument('-heads', type=int, default=8)
    main_arg_parser.add_argument('-dropout', type=int, default=0.1)

    args = main_arg_parser.parse_args()
    main(args)
    end = time.time()
    print("total time: {} minutes\n".format((end - start)/60))

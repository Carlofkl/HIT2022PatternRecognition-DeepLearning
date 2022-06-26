import torch
import torch.nn as nn
import numpy as np


class RNN(nn.Module):
    def __init__(self, args, output_size):
        super(RNN, self).__init__()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.input_size = args.input_size
        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):  # x是一个句子，tensor
        global output
        if not hidden:
            hidden = torch.zeros(1, self.hidden_size).to(self.device)
        x = x[0]
        for i in range(x.shape[0]):
            token = x[i: i + 1]
            combined = torch.cat((token, hidden), 1)
            hidden = self.tanh(self.i2h(combined))
            output = self.h2o(hidden)
        return output


class LSTM(nn.Module):
    def __init__(self, args, output_size,  bidirectional=False):
        super(LSTM, self).__init__()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.input_size = args.input_size
        self.bidirectional = bidirectional
        self.forget_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.input_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.c_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.output_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        if not bidirectional:
            self.h2o = nn.Linear(self.hidden_size, output_size)
        else:
            self.h2o = nn.Linear(self.hidden_size * 2, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global ct, result
        x = x[0]
        if not self.bidirectional:
            hidden = torch.zeros(1, self.hidden_size).to(self.device)
            ct = torch.zeros(1, self.hidden_size).to(self.device)
            for i in range(x.shape[0]):
                token = x[i: i + 1]
                combined = torch.cat((token, hidden), 1)
                forget = self.sigmoid(self.forget_gate(combined))
                input = self.sigmoid(self.input_gate(combined))
                c_ = self.tanh(self.c_gate(combined))
                output = self.sigmoid(self.output_gate(combined))
                ct = ct * forget + input * c_
                hidden = self.tanh(ct) * output
                result = self.h2o(hidden)
            return result
        else:
            num = x.shape[0]
            hidden1 = torch.zeros(1, self.hidden_size).to(self.device)
            hidden2 = torch.zeros(1, self.hidden_size).to(self.device)
            hidden1s = []
            hidden2s = []
            ct1 = torch.zeros(1, self.hidden_size).to(self.device)
            ct2 = torch.zeros(1, self.hidden_size).to(self.device)
            for i in range(num):
                token1 = x[i: i+1]
                token2 = x[num-i-1: num-i]
                combined1 = torch.cat((token1, hidden1), 1)
                combined2 = torch.cat((token2, hidden2), 1)
                forget1 = self.sigmoid(self.forget_gate(combined1))
                forget2 = self.sigmoid(self.forget_gate(combined2))
                input1 = self.sigmoid(self.input_gate(combined1))
                input2 = self.sigmoid(self.input_gate(combined2))
                c_1 = self.tanh(self.c_gate(combined1))
                c_2 = self.tanh(self.c_gate(combined2))
                output1 = self.sigmoid(self.output_gate(combined1))
                output2 = self.sigmoid(self.output_gate(combined2))
                ct1 = ct1 * forget1 + input1 * c_1
                ct2 = ct2 * forget2 + input2 * c_2
                hidden1 = self.tanh(ct1) * output1
                hidden2 = self.tanh(ct2) * output2
                hidden1s.append(hidden1)
                hidden2s.insert(0, hidden2)
            hidden1 = torch.stack(hidden1s).mean(0)
            hidden2 = torch.stack(hidden2s).mean(0)
            result = self.h2o(torch.cat((hidden1, hidden2), 1))
            return result


class GRU(nn.Module):
    def __init__(self, args, output_size):
        super(GRU, self).__init__()
        self.device = args.device
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.reset_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.update_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.h_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global output
        hidden = torch.zeros(1, self.hidden_size).to(self.device)
        ones = torch.ones(1, self.hidden_size).to(self.device)
        x = x[0]
        for i in range(x.shape[0]):
            token = x[i: i + 1]
            combined = torch.cat((token, hidden), 1)  # 1 x (128+_)
            reset = self.sigmoid(self.reset_gate(combined))
            zt = self.sigmoid(self.update_gate(combined))
            combined2 = torch.cat((token, reset * hidden), 1)
            h_ = self.tanh(self.h_gate(combined2))
            hidden = zt * hidden + (ones - zt) * h_
            output = self.h2o(hidden)
        return output



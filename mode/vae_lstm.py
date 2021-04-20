import mxnet as mx
from collections import namedtuple
import numpy as np
LSTMCell = namedtuple('LSTMCell_return',['Ct','h'])
prev_param = namedtuple('prev_param',['c','h'])

class vae_lstm:
    def __init__(self,input,seq_len,batch_size,num_hidden,num_layer,num_dim,init_prev,mean_z,ctx):
        '''

        input shape must be (seq_len,batch_size,num_dim)
        ctx must be mx.cpu or mx.gpu()
        
        '''
        super().__init__()
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.num_layer = num_layer
        self.num_dim = num_dim
        self.input = input
        self.seq_len = seq_len
        self.ctx = ctx
        self.init_prev = init_prev
        self.mean_z = mean_z

    #get parameter function
    #get the lstm init satate function
    def get_init_state(self):
        batch_size = self.batch_size
        num_hidden = self.num_hidden
        c = mx.nd.zeros(shape = (batch_size,num_hidden))
        h = mx.nd.zeros(shape = (batch_size,num_hidden))
        return [c,h]

    #lstm cell function
    def lstm_cell(self,indata,prev_param,param,num_hidden,mean_z):
        '''
        
        indata shape must be (batch_size,num_dim)
        prev_param is the c and h of the previous state 
        param type must be a dict
        
        '''
        def _calculate(indata,param,prev_param,num_hidden,mean_z):
            i2h = mx.nd.dot(indata,param.get('i2h_weight')) + param.get('i2h_bias')
            h2h = mx.nd.dot(prev_param.h,param.get('h2h_weight')) + param.get('h2h_bias')
            z2h = mx.nd.dot(mean_z,param.get('z_weight')) + param.get('z_bias')
            return i2h + h2h + z2h
        forget_gate = mx.nd.Activation(data = _calculate(indata,param[0],prev_param,num_hidden,mean_z),act_type = 'sigmoid')
        input_gate = mx.nd.Activation(data = _calculate(indata,param[1],prev_param,num_hidden,mean_z),act_type = 'sigmoid')
        output_gate = mx.nd.Activation(data = _calculate(indata,param[2],prev_param,num_hidden,mean_z),act_type='sigmoid')
        Ct_cite = mx.nd.Activation(data = _calculate(indata,param[3],prev_param,num_hidden,mean_z),act_type = 'tanh')
        Ct = (input_gate * Ct_cite) + (forget_gate * prev_param.c)
        h = output_gate * mx.nd.Activation(data = Ct,act_type = 'tanh')
        return LSTMCell(Ct=Ct,h = h)
    
    #forward calculate function
    def forward(self,parameter):
        input = self.input
        seq_len = self.seq_len
        num_layer = self.num_layer
        Prevstate = prev_param(c = self.init_prev[0], h = self.init_prev[1]) 
        Forward_Hidden = list()
        for seq in range(seq_len):
            x = self.input[seq]
            z = self.mean_z[seq]
            unroll_hidden = list()
            for layer in range(num_layer):
                param = parameter[layer]
                Lstm_State = self.lstm_cell(indata = x,prev_param = Prevstate,param = param,num_hidden = self.num_hidden,mean_z = z)
                Prevstate = prev_param(c = Lstm_State.Ct,h = Lstm_State.h)
                unroll_hidden.append(Lstm_State.h)
            Forward_Hidden.append(unroll_hidden[-1])
        hidden = mx.nd.concat( *Forward_Hidden,dim = 0).reshape(shape = (self.seq_len,self.batch_size,self.num_hidden))
        return hidden


def get_vae_lstm_param(batch_size,num_hidden,num_layer,num_latent,num_dim,ctx):
    params = list()

    def _getparam(shape):
        return mx.nd.random.normal(scale=0.01,shape = shape,ctx = ctx)

    def _getzero(shape):
        return mx.nd.zeros(shape = shape,ctx = ctx)

    def param():
        return {        
                        'i2h_weight' : _getparam(shape = (num_dim,num_hidden)),
                        'i2h_bias' : _getzero(shape = (num_hidden,)),
                        'h2h_weight' : _getparam(shape = (num_hidden,num_hidden)),
                        'h2h_bias' : _getzero(shape = (num_hidden,)),
                        'z_weight' : _getparam(shape = (num_latent,num_hidden)),
                        'z_bias' : _getzero(shape = num_hidden,)
                }

    for _ in range(num_layer):
        parama = list()
        for __ in range(4):
            #get LSTMparam
            w_b = param() #network parameter
            #Attach a gradient
            w_b.get('i2h_weight').attach_grad()
            w_b.get('i2h_bias').attach_grad()
            w_b.get('h2h_weight').attach_grad()
            w_b.get('h2h_bias').attach_grad()
            w_b.get('z_weight').attach_grad()
            w_b.get('z_bias').attach_grad()
            parama.append(w_b)
        params.append(parama)
    return params

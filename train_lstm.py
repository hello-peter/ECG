import mxnet as mx
from mxnet import autograd
import sys
import numpy as np
sys.path.append('./ECG5000')
sys.path.append('./mode')
sys.path.append('./update')
from data import data_iter
from lstm import lstm,get_lstm_param
from softmax import softmax,get_softmax_params
from algorithm import adam,sgd,get_adam_state

class Train:
    def __init__(self,data_iter,num_hidden,num_layer,num_category,epochs,lr,ctx,adam_up = True):

        '''

        lr = 0.01~0.001
        data_iter  gets the label and the sample of the model
        num_category represents several categories
        adam_up stands for whether or not to use the Adam algorithm

        '''

        super().__init__()
        self.data_iter = data_iter
        self.adam_up = adam_up
        self.num_hidden = num_hidden
        self.num_layer = num_layer
        self.ctx = ctx
        self.num_category = num_category
        self.epochs = epochs
        self.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.lr = lr

    def accuracy(self,y_hat, y):
        return ((y_hat.argmax(axis = 2)+1) == y.astype('float32')).mean().asscalar()

    #forward calculate
    def forward(self):
        #get sequance lenth,batch_size and the dimension of this data
        for x,_ in self.data_iter():
            seq_len = x.shape[0]
            batch_size = x.shape[1]
            num_dim = x.shape[2]
        
        #get lstm parameter and All connection layer parameter
        lstm_param =  get_lstm_param(batch_size = batch_size,num_hidden = self.num_hidden,num_layer = self.num_layer,num_dim = num_dim,ctx = self.ctx)
        softmax_params = get_softmax_params(num_dim = self.num_hidden,num_categrey = self.num_category,ctx = self.ctx)

        #Determine whether to update parameters using the Adam algorithm
        #Get state if adam_up == True
        if self.adam_up == True:
            #get lstm state
            lstm_state = list()
            for layer in range(self.num_layer):
                s = list()
                for cell in lstm_param[layer]:
                    s.append(get_adam_state(params = cell,ctx = self.ctx))
                lstm_state.append(s)
            softmax_state = get_adam_state(params = softmax_params,ctx = self.ctx)

        #Gets the initializing cell and h
        c0 = mx.nd.zeros(shape = (batch_size,self.num_hidden),ctx = self.ctx)
        h0 = mx.nd.zeros(shape = (batch_size,self.num_hidden),ctx = self.ctx)

        for time in range(self.epochs):
            for x,y in data_iter():
                with autograd.record():
                    lstm_hidden = lstm(input = x,seq_len = seq_len,batch_size = batch_size,num_hidden = self.num_hidden,
                    num_layer = self.num_layer,num_dim = num_dim,init_prev=[c0,h0],ctx = self.ctx).forward(lstm_param)
                    softmax_out = softmax(input = lstm_hidden,num_categrey = 5,ctx = self.ctx).forward(softmax_params)
                    l = self.loss(softmax_out,y).mean()
                l.backward()
                if self.adam_up == None:
                    sgd(lr = self.lr,batch_size = batch_size,dict_params = softmax_params)
                    for layer in range(self.num_layer):
                        for cell in lstm_param[layer]:
                            sgd(lr = self.lr,batch_size = batch_size,dict_params = cell)
                    
                elif self.adam_up == True:
                    # update lstm layer parameter
                    adam(lr = self.lr,params = softmax_params,state = softmax_state,time = time + 1,ctx = self.ctx).update()
                    for param,states in zip(lstm_param,lstm_state):
                        for cell,state in zip(param,states):
                            adam(lr = self.lr,params = cell,state = state,time = time+1,ctx = self.ctx).update()
                    
                print('loss %f'%l.asscalar())
                acc = self.accuracy(softmax_out,y)
                print('train accuracy : %f'%acc)
                for x,y in data_iter(train = None):
                    lstm_hidden = lstm(input = x,seq_len = seq_len,batch_size = batch_size,num_hidden = self.num_hidden,
                    num_layer = self.num_layer,num_dim = num_dim,init_prev=[c0,h0],ctx = self.ctx).forward(lstm_param)
                    softmax_out = softmax(input = lstm_hidden,num_categrey = 5,ctx = self.ctx).forward(softmax_params)
                    #test_loss = self.loss(softmax_out,y).mean()
                    #print('test loss %s'%test_loss.asscalar())
                    acc = self.accuracy(softmax_out,y)
                    print('test accuray: %f'%acc)
                
#Determine whether or not to use the GPU
def get_gpu():
    try:
        ctx = mx.gpu()
        _ = mx.nd.zeros((1,),ctx)
        return ctx
    except mx.base.MXNetError:
        return mx.cpu()


if __name__ == "__main__":
    ctx = get_gpu()
    num_hidden = 8
    num_layer = 35
    num_category = 5
    epochs = 5
    Train(data_iter,num_hidden = num_hidden,num_layer = num_layer
    ,num_category = num_category, epochs = epochs,lr = 0.005,ctx = ctx
    ,adam_up = None).forward()



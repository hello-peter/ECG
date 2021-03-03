import mxnet as mx
from mxnet import autograd
import sys
import numpy as np
sys.path.append('./ECG5000')
sys.path.append('./mode')
sys.path.append('./update')
from data import data_iter
from vae_lstm import vae_lstm,get_vae_lstm_param
import train_vae
from train_vae import Train_vae
from softmax import softmax,get_softmax_params
from algorithm import adam,sgd,get_adam_state

class Train:
    def __init__(self,data_iter,num_hidden,num_layer,num_category,num_latent,epochs,lr,vae_params,ctx,adam_up = True):

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
        self.num_latent = num_latent
        self.vae_params = vae_params

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
        lstm_param =  get_vae_lstm_param(batch_size = batch_size,num_hidden = self.num_hidden,num_layer = self.num_layer,num_dim = num_dim,ctx = self.ctx)
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
                z = Create_z(input = x,params = self.vae_params,num_latent = self.num_latent,ctx = self.ctx).make_z()
                with autograd.record():
                    lstm_hidden = vae_lstm(input = x,seq_len = seq_len,batch_size = batch_size,num_hidden = self.num_hidden,
                    num_layer = self.num_layer,num_dim = num_dim,mean_z = z,init_prev=[c0,h0],ctx = self.ctx).forward(lstm_param)
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
                    lstm_hidden = vae_lstm(input = x,seq_len = seq_len,batch_size = batch_size,num_hidden = self.num_hidden,
                    num_layer = self.num_layer,mean_z = z,num_dim = num_dim,init_prev=[c0,h0],ctx = self.ctx).forward(lstm_param)
                    softmax_out = softmax(input = lstm_hidden,num_categrey = 5,ctx = self.ctx).forward(softmax_params)
                    #test_loss = self.loss(softmax_out,y).mean()
                    #print('test loss %s'%test_loss.asscalar())
                    acc = self.accuracy(softmax_out,y)
                    print('test accuray: %f'%acc)



#Use vae encoder to create z hidden layer
class Create_z:
    def __init__(self,input,params,num_latent,ctx):
        super().__init__()
        self.batch_size = input.shape[0]
        self.num_dim = input.shape[1]
        self.input = input
        #self.seq_len = input.shape[0]
        self.ctx = ctx
        self.encoder_param = params.get('encoder_params')
        self.mean_z_params = params.get('mean_z_params')
        self.log_var_z_params = params.get('log_var_z_params')
        self.num_latent = num_latent
    
    def make_z(self):
        encoder_output = self.nn(input = self.input,params = self.encoder_param,ctx = self.ctx)
        #z layer
        mean_z = self.z_layer(params = self.mean_z_params,input = encoder_output)
        return mean_z
    
    def z_layer(self,params,input):
        return mx.nd.dot(input,params.get('w')) + params.get('b')
    #nn layer is Used to encoder and decoder 
    def nn(self,params,input,ctx):
        return mx.nd.dot(input,params.get('w')) + params.get('b')

                  
#Determine whether or not to use the GPU
def get_gpu():
    try:
        ctx = mx.gpu()
        _ = mx.nd.zeros((1,),ctx)
        return ctx
    except mx.base.MXNetError:
        return mx.cpu()



'''if __name__ == "__main__":
    ctx = get_gpu()
    num_hidden = 8
    num_layer = 35
    num_category = 5
    epochs = 5
    Train(data_iter,num_hidden = num_hidden,num_layer = num_layer
    ,num_category = num_category, epochs = epochs,lr = 0.005,ctx = ctx
    ,adam_up = None).forward()'''

if __name__ == "__main__":
    ctx = get_gpu()
    vae_num_hidden = 11
    vae_num_latent = 2
    vae_num_layer = 40 #Abandon parameter
    vae_epochs = 10
    lstm_num_hidden = 8
    lstm_num_layer = 20
    lstm_num_category = 5
    lstm_epochs = 5
    
    vae_params = Train_vae(data_iter = data_iter,num_hidden = vae_num_hidden,num_layer = vae_num_layer,
    num_latent = vae_num_latent,epoch = vae_epochs,lr = 0.01,ctx = mx.cpu(),adam_up = True).forward()  

    Train(data_iter,num_hidden = lstm_num_hidden,num_layer = lstm_num_layer
    ,num_category = lstm_num_category, epochs = lstm_epochs,lr = 0.005,vae_params = vae_params,num_latent = vae_num_latent,ctx = ctx
    ,adam_up = False).forward()



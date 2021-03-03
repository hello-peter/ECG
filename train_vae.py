import mxnet as mx
from mxnet import autograd
import sys
import random
sys.path.append('./ECG5000')
sys.path.append('./mode')
sys.path.append('./update')
from data import data_iter
from vae import vae,get_vae_params
from algorithm import adam,sgd,get_adam_state


class Train_vae:
    def __init__(self,data_iter,num_hidden,num_layer,num_latent,epoch,lr,ctx,adam_up = True):
        super().__init__()
        self.data_iter = data_iter
        self.num_hidden = num_hidden
        self.num_layer = num_layer
        self.num_latent = num_latent
        self.epoch = epoch
        self.lr = lr
        self.adam_up = adam_up
        self.ctx = ctx

    def forward(self):
        for x,_ in self.data_iter():
            seq_len = x.shape[0]
            batch_size = x.shape[1]
            num_dim = x.shape[2]

        #get parameters
        vae_param = get_vae_params(batch_size = batch_size,num_hidden = self.num_hidden,num_layer = self.num_layer,num_dim = num_dim,latten_size = self.num_latent,ctx = self.ctx)

        #get adam state if adam_up = True
        if self.adam_up == True:
            
            '''for layer in range(self.num_layer):
                s = list()
                for cell in vae_param.get('encoder_params')[layer]:
                    s.append(get_adam_state(params = cell,ctx = self.ctx))
                encoder_state.append(s)'''
            encoder_state = get_adam_state(vae_param.get('encoder_params'),ctx = self.ctx)
            
            '''for layer in range(self.num_layer):
                s = list()
                for cell in vae_param.get('decoder_params')[layer]:
                    s.append(get_adam_state(params = cell,ctx = self.ctx))
                decoder_state.append(s)'''
            decoder_state = get_adam_state(vae_param.get('decoder_params'),ctx = self.ctx)
            meanz_state = get_adam_state(vae_param.get('mean_z_params'),ctx = self.ctx)
            logvar_state = get_adam_state(vae_param.get('log_var_z_params'),ctx = self.ctx)

        #train mode
        for time in range(self.epoch):
            for x,y in self.data_iter():
                with autograd.record():
                    radom_int = random.randint(0,998)
                    decoder_output,mean_z,log_var_z = vae(input = x[radom_int],params = vae_param,decoder_numhidden = self.num_hidden,num_layer = self.num_layer,num_latent = self.num_latent,ctx = self.ctx).mode()
                    l = vae.loss(obs = decoder_output,actual = x,mu = mean_z,log_sigma = log_var_z)
                l.backward()
                if self.adam_up == False:
                    '''for layer in range(self.num_layer):
                        for cell in vae_param.get('encoder_params')[layer]:
                            sgd(lr = self.lr,batch_size = batch_size,dict_params = cell)
                    for layer in range(self.num_layer):
                        for cell in vae_param.get('decoder_params')[layer]:
                            sgd(lr = self.lr,batch_size = batch_size,dict_params = cell)'''
                    sgd(lr = self.lr,batch_size = batch_size,dict_params = vae_param.get('encoder_params'))
                    sgd(lr = self.lr,batch_size = batch_size,dict_params = vae_param.get('decoder_params'))
                    sgd(lr = self.lr,batch_size = batch_size,dict_params = vae_param.get('mean_z_params'))
                    sgd(lr = self.lr,batch_size = batch_size,dict_params = vae_param.get('log_var_z_params'))
                else:
                    adam(lr = self.lr,params = vae_param.get('encoder_params'),state = encoder_state,time = time + 1,ctx = self.ctx).update()
                    adam(lr = self.lr,params = vae_param.get('decoder_params'),state = decoder_state,time = time + 1,ctx = self.ctx).update()
                    adam(lr = self.lr,params = vae_param.get('mean_z_params'),state = meanz_state,time = time + 1,ctx = self.ctx).update()
                    adam(lr = self.lr,params = vae_param.get('log_var_z_params'),state = logvar_state,time = time + 1,ctx = self.ctx).update()

        return vae_param




#Determine whether to use the GPU
def get_gpu():
    try:
        ctx = mx.gpu()
        _ = mx.nd.zeros((1,),ctx)
        return ctx
    except mx.base.MXNetError:
        return mx.cpu()

'''if __name__ == "__main__":
    ctx = get_gpu()
    num_hidden = 11
    num_latent = 2
    num_layer = 40
    epochs = 10
    params = Train_vae(data_iter = data_iter,num_hidden = num_hidden,num_layer = num_layer,
    num_latent = num_latent,epoch = epochs,lr = 0.01,ctx = mx.cpu(),adam_up = True).forward()  
'''
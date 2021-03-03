import mxnet as mx
import sys
sys.path.append('./lstm')
from lstm import lstm,get_lstm_param
'''

vae mode base on lstm
code base on mxnet

'''

def get_vae_params(batch_size,num_hidden,num_layer,num_dim,latten_size,ctx):
    #get z params
    def get_z_params():
        return {'w':mx.nd.normal(scale = 0.01,shape = (num_hidden,latten_size),ctx = ctx),
                'b':mx.nd.zeros(shape = (latten_size,))}
    def get_encodernn_params():
        return {
                'w':mx.nd.normal(scale = 0.01,shape = (num_dim,num_hidden),ctx = ctx),
                'b':mx.nd.zeros(shape = (num_hidden,))
                }
    def get_decodernn_params():
        return{
                'w':mx.nd.normal(scale = 0.01,shape = (latten_size,num_dim),ctx = ctx),
                'b':mx.nd.zeros(shape = (num_dim,))
                }
    params = {
            'encoder_params' : get_encodernn_params(),
            'decoder_params' : get_decodernn_params(),
            'mean_z_params' : get_z_params(),
            'log_var_z_params' : get_z_params()
            }
    params.get('mean_z_params').get('w').attach_grad()
    params.get('mean_z_params').get('b').attach_grad()
    params.get('log_var_z_params').get('w').attach_grad()
    params.get('log_var_z_params').get('b').attach_grad()
    params.get('encoder_params').get('w').attach_grad()
    params.get('encoder_params').get('b').attach_grad()
    params.get('decoder_params').get('w').attach_grad()
    params.get('decoder_params').get('b').attach_grad()
    return params
    
''''encoder_params' : get_lstm_param(batch_size,num_hidden,num_layer,num_dim,ctx),'''

class vae:
    def __init__(self,input,params,decoder_numhidden,num_layer,num_latent,ctx):
        '''
        
        input shape must be (seq_len,batch_size,num_dim),Otherwise there will be some error

        '''
        super().__init__()
        self.batch_size = input.shape[0]
        self.num_layer = num_layer
        self.num_dim = input.shape[1]
        self.input = input
        self.num_hidden = decoder_numhidden
        #self.seq_len = input.shape[0]
        self.ctx = ctx
        self.encoder_param = params.get('encoder_params')
        self.decoder_param = params.get('decoder_params')
        self.mean_z_params = params.get('mean_z_params')
        self.log_var_z_params = params.get('log_var_z_params')
        self.num_latent = num_latent

    def mode(self):
        ''' 
       
        vae mode : encoder and decoder layer both use lstm algorithm,so this mode can deal sequence data
        like ECG data.

        '''
       
        #encoder layer,encoder_lstm_hidden shape is (seq_len,batch_size,num_dim)
        '''encoder_output = lstm(input = self.input,seq_len = self.seq_len,batch_size = self.batch_size,num_hidden = self.num_hidden,
                            num_layer = self.num_layer,num_dim = self.num_dim,init_prev=[c0,h0],ctx = self.ctx).forward(self.encoder_lstm_param)'''
        encoder_output = self.nn(input = self.input,params = self.encoder_param,ctx = self.ctx)
        #z layer
        mean_z = self.z_layer(params = self.mean_z_params,input = encoder_output)
        log_var_z = self.z_layer(params = self.log_var_z_params,input = encoder_output)
        log_var_z = mx.nd.Activation(data = log_var_z,act_type = 'softrelu')
        sampled_latent = self.sampleing(mean_z = mean_z,log_var_z = log_var_z,shape = (self.batch_size,self.num_latent),ctx = self.ctx)
        
        #decoder layer
        '''decoder_output = lstm(input = sampled_latent,seq_len = self.seq_len,batch_size = self.batch_size,num_hidden = self.num_dim,
                                num_layer = self.num_layer,num_dim = self.num_latent,init_prev = [c1,h1],ctx = self.ctx).forward(self.decoder_lstm_param)'''
        decoder_output = self.nn(input = sampled_latent,params = self.decoder_param,ctx = self.ctx)
        return decoder_output,mean_z,log_var_z
        
    #calculate mean_z or log_var_z
    def z_layer(self,params,input):
        return mx.nd.dot(input,params.get('w')) + params.get('b')

    def sampleing(self,mean_z,log_var_z,shape,ctx):
        epsilon = mx.nd.normal(shape = shape,ctx = ctx)
        return mean_z + mx.nd.exp(log_var_z / 2) + epsilon

    #nn layer is Used to encoder and decoder 
    def nn(self,params,input,ctx):
        return mx.nd.dot(input,params.get('w')) + params.get('b')

    @staticmethod
    def loss(obs, actual,mu, log_sigma):
        '''
        
        calculate vae loss = BCE + KLD
        KLD:
        (Gaussian) Kullback-Leibler divergence KL(q||p), per training example
        0.5 ( Tr[Σ] + <μ,μ> - k - log|Σ| ) = -0.5 \sum ( 1 + 2logσ - μ² - σ² )
        BCE:
        Binary cross-entropy, per training example


        '''
        KL_divergence = -0.5 * mx.nd.sum(mx.nd.square(mu) + mx.nd.square(log_sigma) - mx.nd.log(1e-8 + mx.nd.square(log_sigma)) - 1,-1)
        #print(KL_divergence.shape)
        BCE = -mx.nd.sum(actual * mx.nd.log(mx.nd.clip(data = obs,a_min = 1e-10,a_max = 1e-2)) + (1 - actual) * mx.nd.log(mx.nd.clip(data = 1 - obs,a_min = 1e-10,a_max = 1e-2)),-1)
        #BCE = mx.nd.sum(mx.gluon.loss.SigmoidBCELoss(from_sigmoid = False).hybrid_forward(F = mx.nd,pred = obs,label = actual))
        return mx.nd.mean(KL_divergence + BCE)
#test function
'''
x = mx.nd.normal(scale = 0.01,shape = (10,5,12))
params = get_vae_params(batch_size = 5,num_hidden = 8,num_layer = 2,num_dim = 12,latten_size = 2,ctx = mx.cpu())
print(vae(input = x,params = params,decoder_numhidden = 8,num_layer = 2,num_latent = 2,ctx = mx.cpu()).mode())

'''

import mxnet as mx
from mxnet import autograd

#sgd function
def sgd(lr, batch_size, dict_params):
    """
    
    Mini-batch stochastic gradient descent.
    
    """
    for key in dict_params:
        dict_params.get(key)[:] = dict_params.get(key) - lr * dict_params.get(key).grad/batch_size

 ## Use this function if updated using the Adam algorithm       
def get_adam_state(params,ctx):
    '''

    get adam state function
    The state must be a list,For example:
                        w.shape = (2,4)
                        state = [[mx.nd.zeros(shape = (2,4)),mx.nd.zeros(shape = (2,4))],[mx.nd.zeros(shape = (4,)),mx.nd.zeros(shape = (4,))]]
    
    '''
    sate = list()
    for i in params:
        sate.append([mx.nd.zeros(shape = (params.get(i).shape),ctx = ctx),mx.nd.zeros(shape = (params.get(i).shape),ctx = ctx)])
    return sate


#adamfunction
class adam:
    def __init__(self,lr,params,state,time,ctx):
        '''
        
        params must be a dict
        state is a list
            for example:[[0,1],[2,3]]

        '''
        super().__init__()
        self.lr = lr
        self.params = params
        self.states = state
        self.time = time
        self.ctx = ctx

    #update parameter function
    def update(self):
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        for i,state in zip(self.params,self.states):
            (v,s) = state
            v[:] = beta1 * v + (1 - beta1) * (self.params.get(i).grad)
            s[:] = beta2 * s + (1 - beta2) * (self.params.get(i).grad.square())
            v_bias_corr = v/(1-beta1**self.time)
            s_bias_corr = s/(1-beta2**self.time)
            self.params.get(i)[:] -= self.lr * v_bias_corr / (s_bias_corr.sqrt() + eps)


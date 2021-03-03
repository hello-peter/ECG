import mxnet as mx
from mxnet import autograd

class softmax:
    def __init__(self,input,num_categrey,ctx):
        '''

        the input size must be (batch_size,num_dim)

        '''
        super().__init__()
        self.num_dim = input.shape[1]
        self.input = input
        self.ctx = ctx
        self.num_categrey = num_categrey
    
    def forward(self,params):
        #output = 0
        variable_x = self.input
        w = params.get('w')
        b = params.get('b')
        y = mx.nd.dot(variable_x,w) + b
        return mx.nd.softmax(y)
 


def get_softmax_params(num_dim,num_categrey,ctx):

    def get_w(num_dim):
        return mx.nd.random.normal(scale = 0.01,shape = (num_dim,num_categrey),ctx = ctx)

    def get_b(num_dim):
        return mx.nd.zeros(shape = (num_categrey,),ctx = ctx)

    param = {'w' : get_w(num_dim), 'b' : get_b(num_dim)}
    for i in param:
        param.get(i).attach_grad()
    return param

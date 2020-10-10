import numpy as np


class Relu():
    
    def function(self,Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
        '''
        
        return np.maximum(0,Z)
    
    def jacobian(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x

class Sigmoid():

    def function(self, Z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''

        return 1.0/(1.0+np.exp(-Z))

    def jacobian(self, Z):
        return sigmoid(Z) * (1-sigmoid(Z))

sigmoid = Sigmoid()        
relu = Relu()

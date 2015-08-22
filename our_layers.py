from lasagne import layers
import theano.tensor as T
from our_utils import get_greyscale

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, kernel_size, **kwargs):
        """
        note: kernel_size = stride and kernel is square for simplicity
        kernel_size must be int
        """
        super(Unpool2DLayer, self).__init__(incoming, **kwargs)
        self.kernel_size = kernel_size


    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[2] = input_shape[2] * self.kernel_size
        output_shape[3] = input_shape[3] * self.kernel_size
        return tuple(output_shape)


    def get_output_for(self, input, **kwargs):
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(self.kernel_size, axis = 2).repeat(self.kernel_size, axis = 3)
        

class GreyscaleLayer(layers.Layer):
    """
    This layer calculates the greyscale of the input image
    """
    def __init__(self, incoming, random_greyscale = False, random_seed = 123, **kwargs):
        """
        input for this layer is 4D tensor
        incoming is considered to have shape : (index, channel, height, width)
        """

        super(GreyscaleLayer, self).__init__(incoming, **kwargs)
        
        self.rng = T.shared_randomstreams.RandomStreams(random_seed)
        self.random_greyscale = random_greyscale


    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = 1
        return tuple(output_shape)


    def get_output_for(self, input, deterministic_greyscale = False, **kwargs):
        """ 
        'deterministic_greyscale' is designed to allow using deterministic
        greyscale during validation
        """
        if (not deterministic_greyscale):
            return get_greyscale(input, self.random_greyscale, self.rng)
        return get_greyscale(input, False, self.rng)
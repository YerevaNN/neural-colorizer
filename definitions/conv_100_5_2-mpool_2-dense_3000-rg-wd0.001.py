import theano.tensor as T
import lasagne
import lasagne.layers as layers
import our_layers


def define_model(input_var):
    """
    Here goes the definition of the network
    """
    
    image_size = 32
    conv_filter_count = 100
    conv_filter_size = 5
    pool_size = 2
    n_dense_units = 3000
    
    input = layers.InputLayer(
        shape = (None, 3, image_size, image_size),
        input_var = input_var
    )
    
    greyscale_input = our_layers.GreyscaleLayer(
        incoming = input,
        random_greyscale = True,
    )
    
    conv1 = layers.Conv2DLayer(
        incoming = greyscale_input,
        num_filters = conv_filter_count,
        filter_size = conv_filter_size,
        stride = 1,
        nonlinearity = lasagne.nonlinearities.sigmoid,
    )
    
    pool1 = layers.MaxPool2DLayer(
        incoming = conv1,
        pool_size = pool_size,
        stride = pool_size,
    ) 

    dense1 = layers.DenseLayer(
        incoming =pool1,
        num_units = n_dense_units, 
        nonlinearity = lasagne.nonlinearities.rectify,
    )
    
    pre_unpool1 = layers.DenseLayer(
        incoming = dense1,
        num_units = conv_filter_count * (image_size + conv_filter_size - 1) ** 2 / (pool_size * pool_size),
        nonlinearity = lasagne.nonlinearities.linear,
    )

    pre_unpool1 = layers.ReshapeLayer(
        incoming = pre_unpool1, 
        shape = (input_var.shape[0], conv_filter_count) + ((image_size + conv_filter_size - 1) / 2, (image_size + conv_filter_size - 1) / 2),
    )
    
    unpool1 = our_layers.Unpool2DLayer(
        incoming = pre_unpool1,
        kernel_size = 2,
    )

    deconv1 = layers.Conv2DLayer(
        incoming = unpool1,
        num_filters = 3,
        filter_size = conv_filter_size,
        stride = 1,
        nonlinearity = lasagne.nonlinearities.sigmoid,
    )
  
    output = layers.ReshapeLayer(
        incoming = deconv1,
        shape = input_var.shape
    )
    
    return output


def get_cost_updates(network, input_var, learning_rate):

    output = layers.get_output(network)
    params = layers.get_all_params(network, trainable = True)

    batch_size = input_var.shape[0]
    flat_input = input_var.reshape((batch_size, 3072))
    flat_output= output.reshape((batch_size, 3072))
    
    # cross entrophy loss
    losses = -T.sum(flat_input * T.log(flat_output) + (1 - flat_input) * T.log(1 - flat_output), axis = 1) 
     
    # add saturation loss
    #saturation = -T.sum(T.std(T.reshape(flat_output, (batch_size, 3, 1024)), axis = 1), axis = 1)
    #losses = losses + 0.2 * saturation

    cost = T.mean(losses)
    
    # add weight decay
    cost = cost + 0.001 * lasagne.regularization.regularize_network_params(
        layer = network,
        penalty = lasagne.regularization.l2,
    )
    
    gradients = T.grad(cost, params)

    # stochastic gradient descent
    #updates = [
    #    (param, param - learning_rate * gradient)
    #    for param, gradient in zip(params, gradients)
    #]
    
    # rmsprop
    #updates = lasagne.updates.rmsprop(gradients, params, learning_rate = learning_rate)
    
    # momentum
    updates = lasagne.updates.nesterov_momentum(gradients, params, learning_rate = learning_rate) 
    
    return (cost, updates)
    
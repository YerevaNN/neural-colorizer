import os
import sys
import time
import imp

import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers as layers

import our_utils
from load_data import load_cifar_train_data, load_cifar_val_data
from deeplearningnet_utils import tile_raster_images

network_name = sys.argv[1]
definition = imp.load_source('definition', 'definitions/%s.py' % network_name)

def test_convae(batch_size = 100, 
                n_epochs = 10000,
                learning_rate = 0.01,
                save_iter = 50,
                info_iter = 20,
                val_iter = 10,
                log_dir = 'logs',
                info_dir = 'plots',
                save_dir = 'models',
                load_dir = 'models',
                load_model_name = '',
                ):
    
    # loading dataset
    train_set_x = load_cifar_train_data()[0].astype(theano.config.floatX)
    train_set_x = theano.shared(train_set_x, borrow = True)
    
    val_set_x = load_cifar_val_data()[0].astype(theano.config.floatX)
    val_set_x = theano.shared(val_set_x, borrow = True)
    print "==> Dataset is loaded"


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] / batch_size
    n_val_batches = val_set_x.get_value(borrow = True).shape[0] / batch_size
    print "==> Number of train batches: %d" % n_train_batches
    print "==> Number of val batches: %d" % n_val_batches


    # create input and target variables
    # input_var and target_var are 4 dimentional tensors: (index, channel, height, width)
    input_var = T.tensor4('X') 
    target_var = T.tensor4('Y')
    index = T.lscalar('index')
    

    # build network
    network = definition.define_model(input_var)
    target_var = layers.get_output(network)

    
    # some functions
    cost, updates = definition.get_cost_updates(
        network = network,
        input_var = input_var,
        learning_rate = learning_rate,
    )

    train = theano.function(
        [index],
        cost,
        updates = updates,
        givens = {
            input_var: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validation = theano.function(
        [index],
        cost,
        givens = {
            input_var: val_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    forward = theano.function(
        [input_var], 
        target_var,
    )
    
    # loading model
    if load_model_name == '':
        start_epoch = 0
    else:
        res = our_utils.load_model(
            network = network, 
            file_name = load_model_name,
            directory = load_dir,
        )
        start_epoch = res['epoch'] + 1
        # learning_rate = res['learning_rate']


    # model_name
    model_name = network_name + "-lrcoef0.8" + "-lr" + str(learning_rate)
    print "==> network_name = %s" % network_name
    print "==> model_name = %s" % model_name
    
    
    # create log files
    if (not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    
    train_loss_file = open(log_dir + '/log_train_' + model_name + '.csv', 'w')
    val_losss_file = open(log_dir + '/log_val_' + model_name + '.csv', 'w')
    
    
    # train     
    print "==> Training has started"
    start_time = time.time()
    for epoch in xrange(start_epoch, n_epochs):
         
        costs = []
        for batch_index in xrange(n_train_batches):
            costs.append(train(batch_index))

        print >> train_loss_file, "%d, %f" % (epoch, np.mean(costs))        
        train_loss_file.flush()
        print "Training epoch %d took %.0fs, lr=%f, loss=%f" % (epoch, time.time() - start_time, learning_rate, np.mean(costs))

        if (epoch == 5):
            learning_rate *= 10

        if (epoch % 20 == 1):
                learning_rate = learning_rate * 0.8
        
        # validation
        if (epoch % val_iter == 1):
            costs = []
            for batch_index in xrange(n_val_batches):
                costs.append(validation(batch_index))
            
            print >> val_losss_file, "%d %f" % (epoch, np.mean(costs))
            val_losss_file.flush()
            print "==> Validation loss = %f" % np.mean(costs)


        # save
        if (epoch % save_iter == 1 and epoch > start_epoch + 1):
            our_utils.save_model(
                network = network,
                epoch = epoch,
                model_name = model_name,
                learning_rate = learning_rate,
                directory = save_dir,
            )
            
    
        # info        
        if (epoch % info_iter == 1):
            our_utils.print_samples(
                images = T.concatenate([train_set_x[0:75], val_set_x[0:75]], axis = 0),
                forward = forward, 
                model_name = model_name,
                epoch = epoch,
                columns = 5,
                directory = info_dir,
            )
            
        
        start_time = time.time()

if __name__ == '__main__':
    if (len(sys.argv) <= 1):
        sys.exit("Usage: convae.py <definition_file_name>")
    
    test_convae(
        batch_size = 100,
        learning_rate = 0.0002,
        load_model_name = '',
        info_iter = 20,
        val_iter = 10,
        save_iter = 100,
    )
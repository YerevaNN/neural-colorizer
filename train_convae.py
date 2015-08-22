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
    # target_var is output of validation network and is defined in definition file
    # output for train network is always layers.get_output(network)
    # these two are seperated, because train and val networks may differ from each other
    network, target_var = definition.define_model(input_var)
    
    
    # learning rate, constant 0.001 if not specified
    if (hasattr(definition, 'lr_policy')):
        lr_policy = definition.lr_policy
        if (lr_policy == 'constant'):
            lr_base = definition.lr_base
        elif (lr_policy == 'geom'):
            lr_base = definition.lr_base
            lr_step = definition.lr_step
            lr_coef = definition.lr_coef
        else:
            raise Exception("Unknown lr_policy\nSupported: 'constant', 'geom'\n")
    else:
        lr_policy = 'constant'
        lr_base = 0.001
    
    learning_rate = theano.shared(np.float32(lr_base))
        
    print "==> lr_policy = %s" % lr_policy 
    if (lr_policy == 'constant'):
        print "\tbase_lr = %f" % lr_base
        lr_desc = "-lr_c%f" % lr_base
    elif (lr_policy == 'geom'):
        print "\tlr_base = %f" % lr_base
        print "\tlr_step = %d" % lr_step
        print "\tlr_coef = %f" % lr_coef
        lr_desc = "-lr_g_%f_%d_%f" % (lr_base, lr_step, lr_coef)

    
    # functions for train network
    train_cost, train_updates = definition.get_cost_updates(
        network = network,
        input_var = input_var,
        output = layers.get_output(network),
        learning_rate = learning_rate,
    )

    train = theano.function(
        [index],
        train_cost,
        updates = train_updates,
        givens = {
            input_var: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    
    # functions for validation network
    # val_updates is a dummy variable 
    forward = theano.function(
        [input_var], 
        target_var,
    )
    
    val_cost, val_updates = definition.get_cost_updates(
        network = network,
        input_var = input_var,
        output = target_var,
        learning_rate = learning_rate,
    )
    
    validation = theano.function(
        [index],
        val_cost,
        givens = {
            input_var: val_set_x[index * batch_size: (index + 1) * batch_size]
        }
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
    model_name = network_name + lr_desc
    if (len(sys.argv) >= 3):
        # to allow run temporary models not changing log files
        model_name = sys.argv[2]
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

    try:
        for epoch in xrange(start_epoch, n_epochs):
             
            costs = []
            for batch_index in xrange(n_train_batches):
                costs.append(train(batch_index))
    
            print >> train_loss_file, "%d, %f" % (epoch, np.mean(costs))        
            train_loss_file.flush()
            print "Training epoch %d took %.0fs, lr=%f, loss=%f" % (epoch, time.time() - start_time, learning_rate.eval(), np.mean(costs))
    
    
            # validation
            if (epoch % val_iter == 1):
                costs = []
                for batch_index in xrange(n_val_batches):
                    costs.append(validation(batch_index))
                
                print >> val_losss_file, "%d %f" % (epoch, np.mean(costs))
                val_losss_file.flush()
                print "==> Validation loss = %f" % np.mean(costs)
    
    
            # learning rate policy
            if (lr_policy == 'geom'):
                if (epoch > start_epoch + 1 and epoch % lr_step == 1):
                    learning_rate *= lr_coef

    
            # save
            if (epoch % save_iter == 1 and epoch > start_epoch + 1):
                our_utils.save_model(
                    network = network,
                    epoch = epoch,
                    model_name = model_name,
                    learning_rate = learning_rate.eval(),
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
                our_utils.plot_filters(
                    filters = layers.get_all_param_values(network)[0],
                    model_name = model_name,
                    epoch = epoch,
                    repeat = 10,
                    columns = 8,
                    directory = info_dir,
                )
            
            start_time = time.time()

    except KeyboardInterrupt:
        answer = raw_input("\nWould you like to save the model ? (y / n) ")
        if (answer in ["y", "Y", "yes"]):
            our_utils.save_model(
                network = network,
                epoch = epoch,
                model_name = model_name,
                learning_rate = learning_rate.eval(),
                directory = save_dir,
            )    
        
        exit()


if __name__ == '__main__':
    if (len(sys.argv) <= 1):
        sys.exit("Usage: convae.py <definition_file_name> [<model_name>]")
    
    test_convae(
        batch_size = 100,
        load_model_name = '',
        info_iter = 20,
        val_iter = 10,
        save_iter = 100,
    )
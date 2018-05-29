from collections import namedtuple
import math
import time
import mxnet as mx
import numpy as np
import sys,os

def modelCNN(X_train,X_test,y_train,y_test):
    sentence_size = X_train.shape[0]
    print(sentence_size)
    vocab_size = sentence_size
    batch_size = 50
    print('batch size', batch_size)

    input_x = mx.sym.Variable('data')  # placeholder for input data
    input_y = mx.sym.Variable('softmax_label')  # placeholder for output label
    num_embed = 300  # dimensions to embed words into
    print('embedding dimensions', num_embed)
    embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
    # reshape embedded data for next layer
    conv_input = mx.sym.Reshape(data=embed_layer, shape=(batch_size, 1, sentence_size, num_embed))

    # create convolution + (max) pooling layer for each filter operation
    filter_list = [3, 4, 5]  # the size of filters to use
    print('convolution filters', filter_list)

    num_filter = 100
    pooled_outputs = []
    for filter_size in filter_list:
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1, 1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)

    # reshape for next layer
    h_pool = mx.sym.Reshape(data=concat, shape=(batch_size, total_filters))

    # dropout layer
    dropout = 0.5
    print('dropout probability', dropout)

    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

        # fully connected layer
    num_label = 2

    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    # softmax output
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    # set CNN pointer to the "back" of the network
    cnn = sm

    # Define the structure of our CNN Model (as a named tuple)
    CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

    # Define what device to train/test on, use GPU if available
    ctx = mx.gpu(1) if mx.test_utils.list_gpus() else mx.cpu()

    arg_names = cnn.list_arguments()

    input_shapes = {}
    input_shapes['data'] = (batch_size, sentence_size)

    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if name in ['softmax_label', 'data']:  # input, output
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
    initializer = mx.initializer.Uniform(0.1)
    for i, name in enumerate(arg_names):
        if name in ['softmax_label', 'data']:  # input, output
            continue
        initializer(mx.init.InitDesc(name), arg_dict[name])

        param_blocks.append((i, arg_dict[name], args_grad[name], name))

    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']

    cnn_model = CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)

    '''
    Train the cnn_model using back prop
    '''

    optimizer = 'rmsprop'
    max_grad_norm = 5.0
    learning_rate = 0.0005
    epoch = 50

    print('optimizer', optimizer)
    print('maximum gradient', max_grad_norm)
    print('learning rate (step size)', learning_rate)
    print('epochs to train for', epoch)

    # create optimizer
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate

    updater = mx.optimizer.get_updater(opt)
    num_total = 0
    # For each training epoch
    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0

        # Over each batch of training data
        for begin in range(0, X_train.shape[0], batch_size):
            batchX = X_train[begin:begin + batch_size]
            batchY = y_train[begin:begin + batch_size]
            if batchX.shape[0] != batch_size:
                continue

            cnn_model.data[:] = batchX
            cnn_model.label[:] = batchY

            # forward
            cnn_model.cnn_exec.forward(is_train=True)

            # backward
            cnn_model.cnn_exec.backward()

            # eval on training data
            num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)


            # update weights
            norm = 0
            for idx, weight, grad, name in cnn_model.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in cnn_model.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)

                updater(idx, grad, weight)

                # reset gradient to zero
                grad[:] = 0.0

        # Decay learning rate for this epoch to ensure we are not "overshooting" optima
        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            print('reset learning rate to %g' % opt.lr)

        # End of training loop for this epoch
        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)

        # Saving checkpoint to disk
        # if (iteration + 1) % 10 == 0:
        # prefix = 'cnn'
        # cnn_model.symbol.save('./%s-symbol.json' % prefix)
        # save_dict = {('arg:%s' % k): v for k, v in cnn_model.cnn_exec.arg_dict.items()}
        # save_dict.update({('aux:%s' % k): v for k, v in cnn_model.cnn_exec.aux_dict.items()})
        # param_name = './%s-%04d.params' % (prefix, iteration)
        # mx.nd.save(param_name, save_dict)
        # print('Saved checkpoint to %s' % param_name)

        # Evaluate model after this epoch on dev (test) set
        num_correct = 0
        num_total = 0

        # For each test batch
        for begin in range(0, X_test.shape[0], batch_size):
            batchX = X_test[begin:begin + batch_size]
            batchY = y_test[begin:begin + batch_size]

            if batchX.shape[0] != batch_size:
                continue

            cnn_model.data[:] = batchX
            cnn_model.cnn_exec.forward(is_train=False)

            num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        dev_acc = num_correct * 100 / float(num_total)
        print('Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
                --- Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc))
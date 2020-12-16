import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import sys, re, cPickle, random, logging, argparse

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from process_data import WordVecs


logger = logging.getLogger("qa.sent.cnn")

class QALeNetConvPoolLayer(object):
    """ Convolution Layer and Pool Layer for Question and Sentence pair """

    def __init__(self, rng, linp, rinp, filter_shape, poolsize):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type linp: theano.tensor.TensorType
        :param linp: symbolic variable that describes the left input of the
        architecture (one minibatch)
        
        :type rinp: theano.tensor.TensorType
        :param rinp: symbolic variable that describes the right input of the
        architecture (one minibatch)

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, 1,
                              filter height,filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.linp = linp
        self.rinp = rinp
        self.filter_shape = filter_shape
        self.poolsize = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),borrow=True,name="W_conv")   
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        
        # convolve input feature maps with filters
        lconv_out = conv.conv2d(input=linp, filters=self.W)
        rconv_out = conv.conv2d(input=rinp, filters=self.W)
        lconv_out_tanh = T.tanh(lconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        rconv_out_tanh = T.tanh(rconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.loutput = downsample.max_pool_2d(input=lconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="average_exc_pad")
        self.routput = downsample.max_pool_2d(input=rconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="average_exc_pad")
        self.params = [self.W, self.b]
        
    def predict(self, lnew_data, rnew_data):
        """
        predict for new data
        """
        lconv_out = conv.conv2d(input=lnew_data, filters=self.W)
        rconv_out = conv.conv2d(input=rnew_data, filters=self.W)
        lconv_out_tanh = T.tanh(lconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        rconv_out_tanh = T.tanh(rconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        loutput = downsample.max_pool_2d(input=lconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="average_exc_pad")
        routput = downsample.max_pool_2d(input=rconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="average_exc_pad")
        return loutput, routput

class BilinearLR(object):
    """
    Bilinear Formed Logistic Regression Class
    """

    def __init__(self, linp, rinp, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type linp: theano.tensor.TensorType
        :param linp: symbolic variable that describes the left input of the
        architecture (one minibatch)
        
        :type rinp: theano.tensor.TensorType
        :param rinp: symbolic variable that describes the right input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of left input units
        
        :type n_out: int
        :param n_out: number of right input units

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out),
                    dtype=theano.config.floatX), # not sure should randomize the weights or not
                    name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(value=0., name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.sigmoid(T.dot(T.dot(linp, self.W), rinp.T).diagonal() + self.b)

        # parameters of the model
        self.params = [self.W, self.b]

    def predict(self, ldata, rdata):
        p_y_given_x = T.nnet.sigmoid(T.dot(T.dot(ldata, self.W), rdata.T).diagonal() + self.b)
        return p_y_given_x

    def get_cost(self, y):
        # cross-entropy loss
        L = - T.mean(y * T.log(self.p_y_given_x) + (1 - y) * T.log(1 - self.p_y_given_x))
        return L


def train_qacnn(datasets, 
        U,                       # pre-trained word embeddings
        filter_hs=[2],           # filter width
        hidden_units=[100,2],   
        shuffle_batch=True, 
        n_epochs=25, 
        lam=0, 
        batch_size=20, 
        lr_decay = 0.95,          # for AdaDelta
        sqr_norm_lim=9):          # for optimization
    """
    return: a list of dicts of lists, each list contains (ansId, groundTruth, prediction) for a question
    """
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0])-3) / 2 
    img_w = U.shape[1]
    lsize, rsize = img_h, img_h
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w), ("filter shape",filter_shapes), 
                  ("hidden_units",hidden_units), ("batch_size",batch_size), 
                  ("lambda",lam), ("learn_decay",lr_decay), 
                  ("sqr_norm_lim",sqr_norm_lim), ("shuffle_batch",shuffle_batch)]
    print parameters    

    # define model architecture
    index = T.lscalar()
    lx = T.matrix('lx')
    rx = T.matrix('rx')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    llayer0_input = Words[T.cast(lx.flatten(),dtype="int32")].reshape((lx.shape[0],1,lx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch
    rlayer0_input = Words[T.cast(rx.flatten(),dtype="int32")].reshape((rx.shape[0],1,rx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch


    conv_layers = []        # layer number = filter number
    llayer1_inputs = []      # layer number = filter number
    rlayer1_inputs = []      # layer number = filter number
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = QALeNetConvPoolLayer(rng, linp=llayer0_input, rinp=rlayer0_input,
                                filter_shape=filter_shape, poolsize=pool_size)
        llayer1_input = conv_layer.loutput.flatten(2)
        rlayer1_input = conv_layer.routput.flatten(2)
        conv_layers.append(conv_layer)
        llayer1_inputs.append(llayer1_input)
        rlayer1_inputs.append(rlayer1_input)
    llayer1_input = T.concatenate(llayer1_inputs,1) # concatenate representations of different filters
    rlayer1_input = T.concatenate(rlayer1_inputs,1) # concatenate representations of different filters
    hidden_units[0] = feature_maps*len(filter_hs)    

    classifier = BilinearLR(llayer1_input, rlayer1_input, hidden_units[0], hidden_units[0])
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    L2_sqr = 0.
    for param in params: L2_sqr += (param ** 2).sum()
    cost = classifier.get_cost(y) + lam * L2_sqr
    grad_updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)

    # shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    # extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_train_batches = new_data.shape[0]/batch_size

    train_set, train_set_orig, val_set, test_set = new_data, datasets[0], datasets[1], datasets[2]
    train_set_lx = theano.shared(np.asarray(train_set[:, :lsize],
                                    dtype=theano.config.floatX),borrow=True)
    train_set_rx = theano.shared(np.asarray(train_set[:, lsize:lsize+rsize],
                                    dtype=theano.config.floatX),borrow=True)
    train_set_y = theano.shared(np.asarray(train_set[:,-1],
                                    dtype="int32"),borrow=True)

    train_set_lx_orig, train_set_rx_orig, train_set_qid_orig, train_set_aid_orig, train_set_y_orig = train_set_orig[:, :lsize], train_set_orig[:, lsize:lsize+rsize], np.asarray(train_set_orig[:,-3], dtype="int32"), np.asarray(train_set_orig[:,-2], dtype="int32"), np.asarray(train_set_orig[:,-1], dtype="int32")
    val_set_lx, val_set_rx, val_set_qid, val_set_aid, val_set_y = val_set[:, :lsize], val_set[:, lsize:lsize+rsize], np.asarray(val_set[:,-3], dtype="int32"), np.asarray(val_set[:,-2], dtype="int32"), np.asarray(val_set[:,-1], dtype="int32")
    test_set_lx, test_set_rx, test_set_qid, test_set_aid, test_set_y = test_set[:, :lsize], test_set[:, lsize:lsize+rsize], np.asarray(test_set[:,-3], dtype="int32"), np.asarray(test_set[:,-2], dtype="int32"), np.asarray(test_set[:,-1], dtype="int32")
    
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            lx: train_set_lx[index*batch_size:(index+1)*batch_size],
            rx: train_set_rx[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})

    test_lpred_layers = []
    test_rpred_layers = []
    test_llayer0_input = Words[T.cast(lx.flatten(),dtype="int32")].reshape((lx.shape[0],1,img_h,Words.shape[1]))
    test_rlayer0_input = Words[T.cast(rx.flatten(),dtype="int32")].reshape((rx.shape[0],1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_llayer0_output, test_rlayer0_output = conv_layer.predict(test_llayer0_input, test_rlayer0_input)
        test_lpred_layers.append(test_llayer0_output.flatten(2))
        test_rpred_layers.append(test_rlayer0_output.flatten(2))
    test_llayer1_input = T.concatenate(test_lpred_layers, 1)
    test_rlayer1_input = T.concatenate(test_rpred_layers, 1)
    test_y_pred = classifier.predict(test_llayer1_input, test_rlayer1_input)
    test_model = theano.function([lx, rx], test_y_pred)   

    #start training over mini-batches
    print '... training'
    epoch = 0
    cost_epoch = 0    
    train_preds_epos, dev_preds_epos, test_preds_epos = [], [], []
    while (epoch < n_epochs):        
        epoch = epoch + 1
        total_cost = 0
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                total_cost += cost_epoch
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                total_cost += cost_epoch
        print "epoch = %d, cost = %f" %(epoch, total_cost)

        train_preds, dev_preds, test_preds = defaultdict(list), defaultdict(list), defaultdict(list)
        ypred = test_model(train_set_lx_orig, train_set_rx_orig)
        for i, pr in enumerate(ypred):
            qid, aid, y = train_set_qid_orig[i], train_set_aid_orig[i], train_set_y_orig[i]
            train_preds[qid].append((aid, y, pr))
        ypred = test_model(val_set_lx, val_set_rx)
        for i, pr in enumerate(ypred):
            qid, aid, y = val_set_qid[i], val_set_aid[i], val_set_y[i]
            dev_preds[qid].append((aid, y, pr))
        ypred = test_model(test_set_lx, test_set_rx)
        for i, pr in enumerate(ypred):
            qid, aid, y = test_set_qid[i], test_set_aid[i], test_set_y[i]
            test_preds[qid].append((aid, y, pr))
        train_preds_epos.append(train_preds)
        dev_preds_epos.append(dev_preds)
        test_preds_epos.append(test_preds)
    return train_preds_epos, dev_preds_epos, test_preds_epos

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def get_idx_from_sent(sent, word_idx_map, max_l=50, filter_h=3):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for i, word in enumerate(words):
        if i >= max_l: break
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_cnn_data(revs, word_idx_map, max_l=50, filter_h=3, val_test_splits=[2,3]):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    val_split, test_split = val_test_splits
    for rev in revs:
        sent = get_idx_from_sent(rev["question"], word_idx_map, max_l, filter_h)
        sent += get_idx_from_sent(rev["answer"], word_idx_map, max_l, filter_h)
        sent.append(rev["qid"])
        sent.append(rev["aid"])
        sent.append(rev["y"])
        if rev["split"]==1:
            train.append(sent)
        if rev["split"]==val_split:
            val.append(sent)
        elif rev["split"]==test_split:
            test.append(sent)
    train = np.array(train,dtype="int")
    val = np.array(val,dtype="int")
    test = np.array(test,dtype="int")
    return [train, val, test]


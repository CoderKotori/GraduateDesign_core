from rnn_layers import *
import init
from Discretization import *
import matplotlib.pyplot as plt


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, input_dim=512, output_dim=128,
                 hidden_dim=128, cell_type='lstm', load_param=None):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.params = {}
        if load_param is not None:
            params = np.load(load_param)
            self.params = params.item()
        else:
            # Initialize parameters for the RNN
            dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
            self.params['Wx'] = np.random.randn(input_dim, dim_mul * hidden_dim)
            self.params['Wx'] /= np.sqrt(input_dim)
            self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
            self.params['Wh'] /= np.sqrt(hidden_dim)
            self.params['b'] = np.zeros(dim_mul * hidden_dim)

            self.params['Wx2'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
            self.params['Wx2'] /= np.sqrt(hidden_dim)
            self.params['Wh2'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
            self.params['Wh2'] /= np.sqrt(hidden_dim)
            self.params['b2'] = np.zeros(dim_mul * hidden_dim)
            # Initialize output to vocab weights
            self.params['W_vocab'] = np.random.randn(hidden_dim, output_dim)
            self.params['W_vocab'] /= np.sqrt(hidden_dim)
            self.params['b_vocab'] = np.zeros(output_dim)

        # Cast parameters to correct dtype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(np.float)

    def loss(self, data_in, data_out=None):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.

        # You'll need this
        # mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        Wx2, Wh2, b2 = self.params['Wx2'], self.params['Wh2'], self.params['b2']
        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        '''first step: compute the initial hidden state h0'''
        h0 = np.zeros((data_in.shape[0], Wh.shape[0]))
        h02 = np.zeros((data_in.shape[0], Wh2.shape[0]))
        # print h0.shape
        '''second step: use a word embedding layer to transform the words in captions_in'''
        # captions_in_embedding, emb_cahce = word_embedding_forward(captions_in, W_embed)
        '''third step: run RNN/LSTM'''
        if self.cell_type == 'rnn':
            h, h_cache = rnn_forward(data_in, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            # notice: here changed
            h1, h1_cache = lstm_forward(data_in, h0, Wx, Wh, b)
            h2, h2_cache = lstm_forward(h1, h02, Wx2, Wh2, b2)
        else:
            raise ValueError('Invalid cell type: "%s"', self.cell_type)
        '''forth step: use wx+b tp compute scores'''
        scores, score_cache = temporal_affine_forward(h2, W_vocab, b_vocab)
        if data_out is None:
            return scores
        '''fifth step: use softmax'''
        loss, dloss = temporal_softmax_loss(scores, data_out, verbose=False)

        '''backward propagation'''
        dscores, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dloss, score_cache)
        if self.cell_type == 'rnn':
            dcaption, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dscores, h_cache)
        elif self.cell_type == 'lstm':
            dh2, dh02, grads['Wx2'], grads['Wh2'], grads['b2'] = lstm_backward(dscores, h2_cache)
            din, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dh2, h1_cache)
        else:
            raise ValueError('Invalid cell type: "%s"', self.cell_type)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def verify(self, data_in, prev_h1, prev_c1, prev_h2, prev_c2, features, out, k=5):
        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        Wx2, Wh2, b2 = self.params['Wx2'], self.params['Wh2'], self.params['b2']
        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        next_h1, next_c1, _ = lstm_step_forward(data_in, prev_h1, prev_c1, Wx, Wh, b)
        next_h2, next_c2, _ = lstm_step_forward(next_h1, prev_h2, prev_c2, Wx2, Wh2, b2)
        scores, _ = temporal_affine_forward(next_h2.reshape((1, 1, -1)), W_vocab, b_vocab)
        N, T, _ = scores.shape
        for i in range(N):
            for t in range(T):
                index = np.argpartition(scores[i, t], -k)[-k:]
                prob_result = features[index]
                if out in prob_result:
                    return next_h1, next_c1, next_h2, next_c2, 0
                else:
                    return next_h1, next_c1, next_h2, next_c2, 1


if __name__ == '__main__':
    from Data import *
    from solver import *

    # init data
    data_str = np.load('files/data_str.npy')
    features = np.load('files/features_normal.npy')
    output_dim = features.shape[0]
    d = Data()
    data = d.load_data()

    num_train = 10000
    seq_length = 10

    pm = np.load('files/disc_pm.npy')
    sp = np.load('files/disc_setpoint.npy')
    pid = np.load('files/pred_pid.npy')
    crc = np.load('files/pred_crcrate.npy')
    ti = np.load('files/pred_timeinterval.npy')

    data[:, d.pressure_measurement] = nearest(pm, data[:, d.pressure_measurement])
    data[:, d.setpoint] = nearest(sp, data[:, d.setpoint])
    data[:, d.crc_rate] = nearest(crc, data[:, d.crc_rate])
    data[:, d.time_interval] = nearest(ti, data[:, d.time_interval])
    data[:, d.gain:d.rate + 1] = nearest_plus(pid, data[:, d.gain:d.rate + 1])
    data = np.concatenate((data[:, d.address:d.time], data[:, d.time_interval].reshape(-1, 1)),
                          axis=1)
    input_dim = data.shape[1]
    data = data.astype(float)
    min_max = []
    for j in range(input_dim):
        min, max = np.nanmin(data[:, j]), np.nanmax(data[:, j])
        # min_max.append([min, max])
        # print min, max
        for i in range(data.shape[0]):
            if np.isnan(data[i, j]):
                data[i, j] = -1.0
            else:
                data[i, j] = (data[i, j] - min) / (max - min)

    # data_in = np.zeros((num_train * seq_length, input_dim))
    # data_out = np.zeros((num_train * seq_length)).astype(object)
    # num = 0
    # result = d.load_data()[:, d.binary_result].astype(int)
    # for it in range(data.shape[0]):
    #     if result[it] == 0 and result[it + 1] == 0 and result[it + 2] == 0:
    #         # add data_in
    #         data_in[num] = data[it]
    #         data_in[num + 1] = data[it + 1]
    #         # add data_out
    #         data_out[num] = data_str[it + 1]
    #         data_out[num + 1] = data_str[it + 2]
    #         num += 2
    #         if num >= num_train * seq_length:
    #             break
    # for i in range(data_out.shape[0]):
    #     pos = np.where(features == data_out[i])[0][0]
    #     # print i, ',', pos
    #     data_out[i] = int(pos)
    # data_out = data_out.astype(int)
    #
    # data_in = data_in.reshape(num_train, seq_length, input_dim)
    # data_out = data_out.reshape(num_train, seq_length)

    data_in, data_out = init.init_input(data, data_str, d.load_data()[:, d.binary_result].astype(int),
                                        output_num=num_train,
                                        length=seq_length)
    data_out = data_out.reshape(-1)
    for i in range(data_out.shape[0]):
        pos = np.where(features == data_out[i])[0][0]
        # print i, ',', pos
        data_out[i] = int(pos)
    data_out = data_out.astype(int)
    data_out = data_out.reshape((num_train, seq_length))

    # for i in range(num_train / 100):
    #     if i == 0:
    #         params_path = None
    #     else:
    #         params_path = 'files/lstm_params.npy'
    #     lstm = CaptioningRNN(input_dim, output_dim, hidden_dim=3096, cell_type='lstm', load_param=params_path)
    #
    #     train_data = {}
    #     train_data['train_in'] = data_in[100 * i:100 * (1 + i)]
    #     train_data['train_out'] = data_out[100 * i:100 * (1 + i)]
    #     solver = Solver(lstm, train_data, update_rule='adam',
    #                     num_epochs=10,
    #                     batch_size=100,
    #                     optim_config={
    #                         'learning_rate': 5e-3,
    #                     },
    #                     lr_decay=0.995,
    #                     verbose=True, print_every=10)
    #     solver.train()
    #     plt.plot(solver.loss_history)
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Loss')
    #     plt.title('Training loss history')
    #     plt.show()
    params_path = 'files/lstm_params.npy'
    lstm = CaptioningRNN(input_dim, output_dim, hidden_dim=3096, cell_type='lstm', load_param=params_path)
    solver = Solver(lstm, {})
    test_data = {}
    test_start = 2000
    test_end = 4000
    test_in = data[test_start:test_end, :]
    test_out = data_str[test_start + 1:test_end + 1]
    test_result = d.load_data()[test_start + 1:test_end + 1, d.binary_result]
    test_in = test_in.reshape(-1, 2, input_dim)
    test_out = test_out.reshape(-1, 2)
    test_result = test_result.reshape(-1, 2)
    # test_in, test_out, test_result = init.lstm_input(data[2000:], data_str[2000:],
    #                                                  d.load_data()[2000:, d.binary_result], output_num=5000)
    test_data['test_in'] = test_in
    test_data['test_out'] = test_out
    test_data['test_result'] = test_result
    for i in range(1, 10):
        tp, tn, fp, fn, count = solver.test(test_data, features)
        print 'k=', i
        print 'true positive: ', tp / count
        print 'true negative: ', tn / count
        print 'false positive: ', fp / count
        print 'false negative: ', fn / count

# -*- coding: utf-8 -*-
import numpy as np

class NeuralNet:
    """ Class for word2vec neural net. """

    def __init__(self, vocab_size, emb_size, learning_rate, num_negative,
        scale):
        """ Initialize input and hidden layers of neural net.

        vocab_size: int. vocabulary size of your corpus or training data
        emb_size: int. word embedding size
        """

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.num_negative = num_negative
        self.wrd_emb = scale*np.random.randn(vocab_size, emb_size)
        self.W = scale*np.random.randn(vocab_size, emb_size)

    def propagate(self, X, sampling_inds):
        """ Runs neural net against a batch of training samples.
        """

        # hidden layer: convert input ids into embedding vectors.
        word_vec = self.wrd_emb[X, :].T
        # output layer
        Z = np.dot(self.W[sampling_inds,:], word_vec)
        # softmax output
        softmax_out = self.softmax(Z)

        return softmax_out, word_vec

    def back_propagate(self, X, Y, softmax_out, word_vec, sampling_inds):
        """Updates weights of neural net using backpropagation with descent
        gradient method.

        Gradients are evaluated using chain rule for partial derivatives.
        Only weights related to a small set of tokens are updated, following
        negative sampling technique.

        Args:
            Y: labels of training data. shape: (vocab_size, batch_size)
            softmax_out: output out of softmax. shape: (vocab_size, batch_size)
        """

        batch_size = word_vec.shape[1]

        dL_dZ = softmax_out - Y[sampling_inds, :]

        #dL_dZ = softmax_out[sampling_inds, :] - Y[sampling_inds, :]
        #dL_dZ = softmax_out
        #dL_dZ[Y.flatten(), np.arange(batch_size)] -= 1

        # evaluate gradients
        dL_dW = (1 / batch_size) * np.dot(dL_dZ, word_vec.T)
        dL_dword_vec = np.dot(self.W.T[:, sampling_inds], dL_dZ)
        #self.W -= self.learning_rate * dL_dW

        # update hidden/output layers
        #dL_dword_vec = np.dot(self.W.T, dL_dZ)
        self.W[sampling_inds, :] -= self.learning_rate * dL_dW
        self.wrd_emb[X,:] -= self.learning_rate * dL_dword_vec.T

    def softmax(self, Z):
        """
        Z: output of dense layer. shape: (vocab_size, batch_size)
        """

        # for stability issues
        x = Z-np.max(Z)
        softmax_out = np.divide(np.exp(x), np.sum(np.exp(x), axis=0, keepdims=True) + 0.001)

        return softmax_out

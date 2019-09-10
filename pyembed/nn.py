# -*- coding: utf-8 -*-
import numpy as np

class NeuralNet:
    """ Class for the word2vec neural net.

    Attributes:
        W1: Matrix with weights of hidden layer.
        W2: Matrix with weights of output layer.

    """

    def __init__(self, vocab_size, emb_size, learning_rate, negative):
        """Initializes input and hidden layers of neural net.

        Args:
            vocab_size: Vocabulary size of your corpus or training data.
            emb_size: Word embedding size.
            learning_rate: Rate for weight update equations.
            negative: If greater than 0, it is the number of negative samples.
                If less or equal to 0, negative sampling is disabled.
        """

        # parameters
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.negative = negative

        # init weight matrices
        self.W1 = 0.01*np.random.randn(vocab_size, emb_size)
        self.W2 = 0.01*np.random.randn(emb_size, vocab_size)

    def forward_propagate(self, input_label, sampling_inds):
        """Performs the forward propagation.

        Args:
            input_label: Label of input word.
            sampling_inds: Labels of negative samples (used if negative
                sampling is applied).

        Returns:
            Numpy array: Output out of softmax (shape: vocab_size).
            Numpy array: Embedding of input word (shape: embedding_size).
        """

        # output from hidden layer
        h_vector = self.W1[input_label, :]

        # evaluate output
        if self.negative > 0:

            # output layer - matrix output
            z_vector = np.dot(h_vector, self.W2[:, sampling_inds])
            # output layer - model output
            y_vector = self.sigmoid(z_vector)

        else:

            # output layer - matrix output
            z_vector = np.dot(h_vector, self.W2)
            # output layer - model output
            y_vector = self.softmax(z_vector)

        return y_vector, h_vector

    def back_propagate(self, input_label, t_vector, y_vector, h_vector,
        sampling_inds):
        """Updates weights of neural net using backpropagation with descent
        gradient method.

        Gradients are evaluated using chain rule for partial derivatives.

        Args:
            input_label: Label of input word.
            t_vector: Labels of training data (shape: vocab_size).
            y_vector: Output out of softmax (shape: vocab_size).
            h_vector: Output from hidden layer.
            sampling_inds: Labels of positive + negative samples (used only if
                negative sampling is applied).
        """

        # evaluate gradients
        dL_dZ = y_vector - t_vector

        if self.negative > 0:

            dL_dW1 = np.dot(self.W2[:, sampling_inds], dL_dZ)
            dL_dW2 = np.outer(dL_dZ, h_vector).T

            # update output matrix
            self.W2[:, sampling_inds] -= self.learning_rate * dL_dW2

        else:

            dL_dW1 = np.dot(self.W2, dL_dZ)
            dL_dW2 = np.outer(dL_dZ, h_vector).T

            # update output matrix
            self.W2 -= self.learning_rate * dL_dW2

        # update hidden matrix (same for both cases)
        self.W1[input_label,:] -= self.learning_rate * dL_dW1

    def sigmoid(self, z_vector):
        """Sigmoid function for output layer.

        Args:
            z_vector: Output W2 matrix (shape: vocab_size, 1).

        Returns:
            Numpy array: The output of model.
        """

        sigmoid_out = 1/(1 + np.exp(-z_vector))

        return sigmoid_out

    def softmax(self, z_vector):
        """Softmax function for output layer.

        Args:
            z_vector: Output W2 matrix (shape: vocab_size, 1).

        Returns:
            Numpy vector: The output of model.
        """

        # for stability issues
        x = z_vector-np.max(z_vector)
        softmax_out = np.divide(np.exp(x), np.sum(np.exp(x), axis=0,
            keepdims=True) + 0.001)

        return softmax_out

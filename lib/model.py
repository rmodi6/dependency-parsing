# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        # raise NotImplementedError

        # Calculate the cubic activation function as x^3
        return tf.math.pow(vector, 3)

        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start

        # TruncatedNormal initializer for initializing weights
        w_init = tf.initializers.TruncatedNormal()
        # Initialize the word-pos-label embeddings, and weights of hidden and output state of the model
        self.embeddings = tf.Variable(initial_value=w_init(shape=(vocab_size, embedding_dim)),
                                      trainable=trainable_embeddings)
        self.W1 = tf.Variable(initial_value=w_init(shape=(hidden_dim, embedding_dim * num_tokens)))
        self.W2 = tf.Variable(initial_value=w_init(shape=(num_transitions, hidden_dim)))

        # Zeros initializer for initializing bias
        b_init = tf.zeros_initializer()
        # Initialize the bias vector for the hidden state
        self.B1 = tf.Variable(initial_value=b_init(shape=(hidden_dim,)))

        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        # Get the batch_size and num_tokens from inputs
        batch_size, num_tokens = inputs.shape

        # Get the embeddings for the inputs
        input_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        # Reshape the embeddings to flatten it out
        input_embed = tf.reshape(input_embed, [batch_size, -1])

        h = tf.matmul(input_embed, self.W1, transpose_b=True) + self.B1  # Calculate the output of the hidden layer h

        h = self._activation(h)  # Apply the activation function

        logits = tf.matmul(h, self.W2, transpose_b=True)  # Calculate the output of the output layer

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        def stable_softmax(x: tf.Tensor) -> tf.Tensor:
            """
            Implementation of the stable softmax function. This method computes the numerically stable softmax
            value for the input tensor x.
            :param x: logits whose softmax is to be computed
            :return: softmax tensor of the input x
            """
            # Subtract the max(x) from x to avoid overflow
            z = x - tf.reduce_max(x, axis=1, keepdims=True)
            # Compute the numerator of softmax function as e^z
            numerator = tf.exp(z)
            # Compute the denominator of softmax function as sum(e^z)
            denominator = tf.reduce_sum(numerator, axis=1, keepdims=True)
            # Return division e^z / sum(e^z)
            return tf.math.divide_no_nan(numerator, denominator)

        # Create a mask to remove the infeasible transitions denoted by -1 in labels
        mask = labels != -1
        mask = tf.cast(mask, dtype='float32')

        # Apply the mask to the logits and the labels
        y_pred = logits * mask
        y = labels * mask

        # Get the softmax of the logits
        p = stable_softmax(y_pred)
        # Set min value to a noise to avoid log(0) NaN
        p = tf.clip_by_value(p, 1e-10, 1.0)
        # Compute the cross entropy loss
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(p), axis=1))

        # Compute the regularization as the sum of l2_norm of all the trainable variables of the model
        regularization = tf.add_n([tf.nn.l2_loss(variable) for variable in self.trainable_variables])
        # Multiply the regularization by the regularization lambda
        regularization = self._regularization_lambda * regularization

        # TODO(Students) End
        return loss + regularization

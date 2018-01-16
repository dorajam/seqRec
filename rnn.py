'''
This is a toy example for 1-layer Recurrent Neural Network (RNN).
The network takes in a 2-dimensional array,
and outputs a hidden state vector of the same dimensions each timestamp t.
'''

import numpy as np


def create_toy_data():
    return np.ones([1,2])


def create_fake_sequential_data():
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    data = np.zeros([26,26])
    for i in xrange(26):
        data[(i,i)] = 1
    return data


def create_weights(size=[5,3]):
    return np.random.randn(size)


def sigmoid_prime(activated):
    '''derivative of activated vector w.r.t. preactivation'''
    return 1*(1-activated)


def sigmoid(pre_activated):
    return np.sigmoid(pre_activated)


def forwardprop(x, h_prev, weight, h_weight):
    '''
    Returns activated hidden state at time t

    :param x: input vector
    :param h_prev: previously activated hidden state, at t=0, this is a vector of zeros.
    :param weight: parameters for input vector x.
    :param h_weight: parameters for previous hidden state h_prev.
    :return:
        sigmoid(dot(x, weight) + dot(h_prev, weight_h))
    '''

    pre_activated = np.dot(x, weight) + np.dot(h_prev, h_weight)
    activated = sigmoid(pre_activated)
    return activated

def loss(y_hat):
    return np.sum((y-y_hat)**2)


def get_gradients(final_activated, x):
    '''
    There are two partial derivatives we need to find:
    - one for the current input vector W, and one for the previous hidden state W_h

    :param final_activated: the result of the final activation at the last timestamp
    :return: returns d(final_activated) w.r.t. weights
    '''

    sig_prime = sigmoid_prime(final_activated)
    preactivated_prime = x
    return sig_prime * preactivated_prime


def get_gradients_for_previous_state(previous_states):
    if len(previous_states) < 2:
        return
    current_h = previous_states[-1]
    prev_h = previous_states[-2]
    grads = get_gradients(current_h, prev_h)
    previous_states.pop()
    grads *= get_gradients_for_previous_state(previous_states)
    return grads


def apply_gradients(weights, gradients, learning_rate=0.01):
    return weights + learning_rate * gradients


def backprop(sequential_inputs, previous_states, weights, h_weights):
    w_grads = get_gradients(previous_states[-1], sequential_inputs[-1])
    hidden_w_grads = get_gradients_for_previous_state(previous_states)
    updates = [[weights, w_grads], [h_weights, hidden_w_grads]]
    return map(apply_gradients, updates)


def run(sequential_inputs, eta=0.01, timestamps=10):
    assert len(sequential_inputs) >= timestamps
    hidden_states = []
    inp_dim = sequential_inputs[0].shape[0]
    output_dim = sequential_inputs[0].shape[0]
    weights = create_weights(size=[inp_dim, output_dim])
    h_weights = create_weights(output_dim, output_dim)

    for i in xrange(timestamps):
        h_prev = 0 if len(hidden_states) == 0 else hidden_states[-1]
        hidden_states.append(forwardprop(sequential_inputs[i], h_prev, weights, h_weights))
        weights, h_weights = backprop(sequential_inputs, previous_states, weights, h_weights)



if __name__ == '__main__':
    sequential_inputs = create_fake_sequential_data()
    run(sequential_inputs)


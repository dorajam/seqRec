# backpropagation helpers for multilayer perceptron - currently only for sigmoid activations

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    ''' Returns the derivative of sigmoid(z = w.x + b) w.r.t. z'''
    return sigmoid(z)*(1-sigmoid(z))


# dC/dy_hat
def mse_prime(y, activation, n):
    ''' Derivative was multiplied by 1/2 for convinience'''
    return 1/n*(activation - y)

# da/dz for sigmoid activations
def activation_prime(z):
    ''' Sigma(pre_activation) w.r.t. pre_activation '''
    return sigmoid_prime(z)

# dz/da
def preactivation_prime(w):
    ''' Here x stands both for the input, and also for the hidden layer activations that will feed into subsequent layers.
    xw + b prime w.r.t. w = x or hw2 + b where h = sigma(xw1 + b) this function returns w2*sigma_prime(z)'''

    # how do you identify when x can be further propagated?
    return w

# dz/dw
def final_preactivation_prime():
    return x


def get_weight_gradients(preactivations, activations, weights, labels):
    ''' Notice the repetative pattern as num of layers increases.
    This allows us to solve this recursively.
    :param pre_activations = input x and list of all z preactivations
    :param activations = list of all a activations
    :param weights = list of all weights
    '''
    if len(weights) == 0:
        return 1

    activation = activations[-1]
    pre_activation = preactivations[-1]
    prev_activation = preactivations[-2]

    # dC/dy_hat * dy_hat/da
    pre_grad = mse_prime(labels, activation) * activation_prime(pre_activation)
    # dC/dy_hat * dy_hat/dz * dz/dw
    final_grads = np.dot(preactivation_prime(prev_activation).transpose(), pre_grad) # dot of (h x n) and (n x number of output) = (h x number of output)

    if not final:
        pre_grad = np.dot(pre_grad, preactivation_prime(w).transpose()) # (n x h) dot (h x f) =>> (n x f)
        pre_grad *= sigmoid_prime(prev_activation) # (n x f) * (n x f)
    else:
        final_grads = np.dot(preactivation_prime(prev_activation).transpose(), pre_grad) # dot of (h x n) and (n x number of output) = (h x number of output)

    return final_grads * get_weight_gradients(preactivations[:-1], activations[:-1], weights[:-1], labels)


def backprop(preactivations, activations, weights, labels):
    weights = {'w%s'%i: weight for i, weight in enumerate(weights)}
    grads_for_w1 = get_weight_gradients(preactivations, activations, weights, labels)

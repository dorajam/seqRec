import os
import shutil

import tensorflow as tf
import numpy as np


class RNN:
    def __init__(self, num_of_features, n_hidden, eta=0.1, lmbd=0.001, seed=11):
        self.num_of_features = num_of_features
        self.n_hidden = n_hidden
        self.eta = eta
        self.lmbd = lmbd
        self.seed = seed
        self.weights = []
        self.graph = None

    def xavier_init(self, size, seed):
        xavier_stddev = 1. / tf.sqrt(size[0] / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev, seed=seed)

    def define_model(self, x, h, labels):
        # parameters
        endpoints = {}
        with tf.variable_scope('rnn'):
            w1 = tf.Variable(self.xavier_init([self.num_of_features, self.n_hidden], self.seed))
            w2 = tf.Variable(self.xavier_init([self.n_hidden, self.num_of_features], self.seed))
            v = tf.Variable(self.xavier_init([self.n_hidden, self.n_hidden], self.seed))
            parameters = [w1,w2,v]

            # ops
            hidden = tf.matmul(x, w1) + tf.matmul(h, v)
            hidden_activated = tf.sigmoid(hidden)
            y_logit = tf.matmul(hidden_activated, w2)
            y_hat = tf.nn.softmax(y_logit)
            endpoints['reg'] = self.lmbd * tf.nn.l2_loss(np.sum([w1]))
            endpoints['parameters'] = parameters
            endpoints['y_logit'] = y_logit
            endpoints['hidden_activated'] = hidden_activated
            return y_hat, endpoints


    def predict(self, data, sequence, model_path='model/', clean=True):
        results, probs = [], []
        with tf.Graph().as_default() as g:

            x = tf.placeholder(tf.float32, shape=[None, self.num_of_features], name='x')
            labels = tf.placeholder(tf.float32, shape=[None, self.num_of_features], name='label')
            h = tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='h_prev')
            y_hat, endpoints = self.define_model(x, h, labels)
            hidden_activated = endpoints['hidden_activated']
            h_prev = np.zeros([1, self.n_hidden], dtype=np.float32)

            checkpoint_path = tf.train.latest_checkpoint(model_path)

            with tf.Session() as sess:
                tf.train.Saver().restore(sess, checkpoint_path)
                for inp in data:
                    prob, h_prev = sess.run([y_hat, hidden_activated], feed_dict={x: inp, h: h_prev})
                    probs.append(np.max(prob))
                    results.append(sequence[np.argmax(prob)])
        if clean: shutil.rmtree(model_path)
        return zip(sequence, results, probs)


    def run_graph(self, data, sequence, timestamps=10, model_path='model/'):
        with tf.Graph().as_default() as g:

            # inputs
            x = tf.placeholder(tf.float32, shape=[None, self.num_of_features], name='x')
            labels = tf.placeholder(tf.float32, shape=[None, self.num_of_features], name='label')
            h = tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='h_prev')

            y_hat, endpoints = self.define_model(x, h, labels)
            parameters = endpoints['parameters']
            reg = endpoints['reg']
            y_logit = endpoints['y_logit']
            hidden_activated = endpoints['hidden_activated']

            # optimizer
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_logit, labels=labels)) + reg
            optimizer = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(loss, var_list=parameters)

            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            h_prev = np.zeros([1, self.n_hidden], dtype=np.float32)

            for t in range(timestamps):
                for j in range(data[:-1].shape[0]):
                    curr = data[j]
                    nxt = data[j+1]
                    _, rnn_loss, h_prev = sess.run([optimizer, loss, hidden_activated], feed_dict={x: curr, labels: nxt, h: h_prev})

                if t % 20 == 0:
#                     print 'loss is: ', rnn_loss
                    test_id = np.random.randint(0,data.shape[0])
                    predicted = sess.run(y_hat, feed_dict={x: data[test_id], h: h_prev})
                    top_three_choices =  list(np.argsort(predicted).ravel()[::-1][:3])
                    res = [sequence[i] for i in top_three_choices]
                    print sequence[test_id], '-->', res, 'with p = ', predicted[0][top_three_choices]
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver.save(sess, os.path.join(model_path, 'rnn'), global_step=i)


if __name__ == '__main__':
    sequence = 'abcdefghijklmnopqrstuvwxyz'
    length = len(sequence)
    data = np.eye(length).reshape((length,1,length))
    n_obs, _, num_features  = data.shape

    rnn = RNN(num_features, n_hidden=60, eta=0.001, lmbd=0.001)
    rnn.run_graph(data, sequence, timestamps=400)
    res = rnn.predict(data, sequence)
    inp_to_target = {e: t for e,t in zip(sequence[:-1], sequence[1:])}
    accuracy = sum([inp_to_target[row[0]] == row[1] for row in res if row[0] != 'z']) / float(length) # hacky
    print accuracy

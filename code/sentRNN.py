import numpy as np
import theano
import theano.tensor as T
import time
import sys
from collections import OrderedDict
from datasets import build_vocab, index_data
from deform_test import disform_sentences
from theano.ifelse import ifelse
import os
import six.moves.cPickle as pickle


sys.setrecursionlimit(1000000)

class SentRNN(object):
    def __init__(self,X=None, n=50, numpy_rng=None, W=None, U=None, V=None, Vindex=None):
        if W is None:
            initial_W = np.asarray(
                    numpy_rng.uniform(
                        low=-4 * np.sqrt(6/n),
                        high=4 * np.sqrt(6/n),
                        size=(1,n)),
                    dtype=theano.config.floatX)
            W = theano.shared(
                    value=initial_W,
                    name='W',
                    borrow=True)

        if U is None:
            initial_U = np.asarray(
                    numpy_rng.uniform(
                        low=-4 * np.sqrt(2/n),
                        high=4 * np.sqrt(2/n),
                        size=(n, 2 * n)),
                    dtype=theano.config.floatX)
            U = theano.shared(
                    value=initial_U,
                    name='U',
                    borrow=True)

        self.W = W

        self.U = U

        self.X = X

        self.V = V

        self.Vindex = Vindex

        self.params = [self.W, self.U, self.X]

        idxs = T.ivector('idxs')
        idxs_dist = T.ivector('idxs')

        x = self.X[idxs]
        x_dist = self.X[idxs_dist]

        def composition_fn(x_t, h_tm1):
            return T.nnet.sigmoid(
                    T.dot(self.U,
                        T.concatenate([x_t, h_tm1])))

        pa, _ = theano.scan(fn=composition_fn,
                            sequences=x[1:],
                            outputs_info=x[0])

        pa_dist, _ = theano.scan(fn=composition_fn,
                                 sequences=x_dist[1:],
                                 outputs_info=x_dist[0])

        embeds_dist_comp = T.concatenate([T.shape_padleft(x_dist[0]),
                                     T.shape_padleft(x_dist[1]),
                                     pa_dist])

        embeds_comp = T.concatenate([T.shape_padleft(x[0]),
                                T.shape_padleft(x[1]),
                                pa])

        embeds_dist = ifelse(T.gt(T.shape(idxs_dist)[0],
                                    T.as_tensor_variable(np.asarray(1))),
                                    embeds_dist_comp, T.concatenate([x_dist, T.zeros((1,n))]))

        embeds = ifelse(T.gt(T.shape(idxs)[0],
                                T.as_tensor_variable(np.asarray(1))),
                                embeds_comp, T.concatenate([x, T.zeros((1,n))]))

        self.embed_fn = theano.function(inputs=[idxs],
                                        outputs=embeds)

        prob, _ = theano.scan(lambda e_i:
                                T.dot(self.W, e_i),
                                outputs_info=None,
                                sequences=embeds)

        prob_dist, _ = theano.scan(lambda e_i:
                                    T.dot(self.W, e_i),
                                    outputs_info=None,
                                    sequences=embeds_dist)

        self.prob_fn = theano.function(inputs=[idxs],
                                        outputs=T.sum(prob))

        lr = T.scalar('lr')
        l2reg = T.scalar('l2reg')

        l2_sqr = ((self.W ** 2).sum() +(self.U ** 2).sum())

        cont_ent = T.sum(prob_dist) - T.sum(prob)

        self.cont_ent_fn = theano.function(inputs=[idxs, idxs_dist],
                                           outputs=cont_ent)

        hinge_loss = T.nnet.relu(1 - cont_ent)

        hinge_loss_grad = T.grad(hinge_loss + l2reg * l2_sqr, self.params)

        hinge_loss_updates = OrderedDict((p, p-lr*g)
                                            for p, g in
                                            zip(self.params, hinge_loss_grad))

        self.hinge_loss_train_fn = theano.function(
                                        inputs=[idxs, idxs_dist, lr, l2reg],
                                        outputs=cont_ent,
                                        updates=hinge_loss_updates)


        nll = -1 * T.sum(prob)

        nll_grad = T.grad(nll + l2reg * l2_sqr, self.params)


        nll_updates = OrderedDict((p, p-lr*g)
                        for p, g in zip(self.params, nll_grad))

        self.nll_train_fn = theano.function(
                                    inputs=[idxs, idxs_dist, lr, l2reg],
                                    outputs=cont_ent,
                                    updates=nll_updates)

        self.normalize = theano.function(inputs=[],
                updates={self.X:
                            self.X/ T.sqrt((self.X ** 2)
                            .sum(axis=1))
                            .dimshuffle(0, 'x')})


    def load(self, folder):
        for param in self.params:
            param.set_value(np.load(os.path.join(folder,
                            param.name + '.npy')))


def train(learning_rate=0.2, n=50, n_epochs=5, dataset_train='../data/ptb.train.txt',
            train_dist=10.0, valid_dist=30.0, dataset_valid='../data/ptb.valid.txt',
            l2reg=0.001, pct=100):

    folder_name = os.path.basename(__file__).split('.')[0]
    folder = os.path.join(os.path.dirname(__file__), folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    dist_train = disform_sentences(dataset_train, 100 - train_dist,
                                    train_dist/2, train_dist/2)

    S_train, V, Vindex = build_vocab(dataset_train)

    S_train_dist = index_data(dist_train, Vindex)

    S_train_pair = zip(S_train, S_train_dist)

    n_train = len(S_train)

    numpy_rng = np.random.RandomState(123)

    V_len = len(V)

    X_initial = np.asarray(
           numpy_rng.uniform(
                low=-4 *np.sqrt(6./V_len),
                high=4 * np.sqrt(6./V_len),
                size=(V_len, n)),
           dtype=theano.config.floatX)

    X = theano.shared(
            value=X_initial,
            name='X',
            borrow=True)

    ###############
    # TRAIN MODEL #
    ###############

    best_validation_ce = -np.inf

    done_looping = False
    epoch = 0

    sentRNN = SentRNN(
            numpy_rng=numpy_rng,
            n=n,
            X=X,
            V=V,
            Vindex=Vindex)

    train_upto_idx = int(pct * n_train/100)
    print '... training the model'
    while (epoch < n_epochs) and (not done_looping):
        tic = time.time()
        epoch = epoch + 1

        np.random.shuffle(S_train_pair)
        te_cost = 0.0
        run_upto_idx = pct * n_train
        for i, sentence_pair in enumerate(S_train_pair):

            if i > train_upto_idx:
                break

            assert len(sentence_pair[0]) == len(sentence_pair[1])
            if len(sentence_pair[0]) < 2:
                continue
            try:
                train_cost = sentRNN.hinge_loss_train_fn(
                                sentence_pair[0],
                                sentence_pair[1],
                                learning_rate/(1 + .001 * epoch),
                                l2reg)

                te_cost += train_cost
                print '[learning embedding] epoch %i >> %2.2f%% completed in %.2f (sec) cost >> %2.2f <<\r' % (
                    epoch, (i + 1) * 100. / n_train, time.time() - tic, te_cost),
                sys.stdout.flush()
            except:
                import pdb;pdb.set_trace()
                print sentence_pair
                raise

        print '[learning embedding] epoch %i >> completed in %.2f (sec) T cost >> %2.2f <<\r' % (
            epoch, time.time() - tic, te_cost/n_train)
        sys.stdout.flush()
        sentRNN.normalize()

        avg_valid_value = 0;
        for j in xrange(10):
            dist_valid = disform_sentences(dataset_valid, 100 - valid_dist,
                                            valid_dist/2, valid_dist/2)

            S_valid = index_data(dataset_valid, Vindex)

            S_valid_dist = index_data(dist_valid, Vindex)

            S_valid_pair = zip(S_valid, S_valid_dist)

            n_valid = len(S_valid)

            tic = time.time()
            total_valid_cost = 0.0

            for i, sentence_pair in enumerate(S_valid_pair):
                if len(sentence_pair[0]) < 2:
                    continue
                valid_cost = sentRNN.cont_ent_fn(
                                sentence_pair[0],
                                sentence_pair[1])
                total_valid_cost += valid_cost
            total_valid_cost /= n_valid
            print '[validating] epoch %i iter %i >> completed in %.2f (sec) Valid cost >> %2.2f <<\r' % (
                epoch, j, time.time() - tic, total_valid_cost)
            sys.stdout.flush()
            avg_valid_value += total_valid_cost
        avg_valid_value /= 10;
        print '[validating] epoch %i >> completed in %.2f (sec) Valid cost >> %2.2f <<\r' % (
            epoch, time.time() - tic, pow(2, avg_valid_value))

        if avg_valid_value > best_validation_ce:
            print '[validation] Updating epoch %i >> bestValidationCE=> Old: %5.4f, New: %5.4f' % (
                    epoch, best_validation_ce, avg_valid_value)
            best_validation_ce = avg_valid_value
            # save the best model
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(sentRNN, f)


def test(model_file=None, err_pct=30, dataset_test='../data/ptb.test.txt'):
    if model_file == None:
        model_file = 'best_model.pkl'

    sentRNN = pickle.load(open(model_file))
    avg_test_value = 0
    for j in xrange(10):
        dist_test = disform_sentences(dataset_test, 100 - err_pct, err_pct/2, err_pct/2)

        S_test = index_data(dataset_test, sentRNN.Vindex)

        S_test_dist = index_data(dist_test, sentRNN.Vindex)

        S_test_pair = zip(S_test, S_test_dist)

        n_test = len(S_test)

        tic = time.time()
        total_test_cost = 0.0
        for i, sentence_pair in enumerate(S_test_pair):
            if len(sentence_pair[0]) < 2:
                continue
            test_cost = sentRNN.cont_ent_fn(
                            sentence_pair[0],
                            sentence_pair[1])
            total_test_cost += test_cost
        total_test_cost /= n_test
        print '[testing] iter %i >> completed in %.2f (sec) test cost >> %2.2f <<\r' % (
            j, time.time() - tic, total_test_cost)
        sys.stdout.flush()
        avg_test_value += total_test_cost
    avg_test_value /= 10;
    print '[testing] completed in %.2f (sec) test cost >> %2.2f <<\r' % (
        time.time() - tic, pow(2, avg_test_value))

if __name__ == '__main__':
    X = theano.shared(
        np.random.uniform(
            high=1,
            low=-1,
            size=(6, 2)))

    numpy_rng = np.random.RandomState(123)

    #sentRNN = SentRNN(n = 2, X=X, numpy_rng=numpy_rng)

    train()
    test()

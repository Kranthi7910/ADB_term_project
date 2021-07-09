import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from tqdm import tqdm
from utils import plot_result
from utils import NNfuncs


class Model(chainer.Chain):
    def __init__(self, n_in, n_units1, n_units2, n_out):
        super(Model, self).__init__(
            l1=L.Linear(n_in, n_units1),
            l2=L.Linear(n_units1, n_units2),
            l3=L.Linear(n_units2, n_out),
        )

    def __call__(self, x, t):
        y = self.l3(F.relu(self.l2(F.relu(self.l1(x)))))
        self.loss = self.jsd(t, y)
        #self.loss = self.listwise_cost(t, y)
        return self.loss

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = F.relu(self.l3(h2))
        return h.data

    def kld(self, vec_true, vec_compare):
        # Kullback–Leibler divergence
        nd = vec_true.data * vec_compare.data > 0
        nd_var = chainer.Variable(nd)
        include_nan = vec_true * F.log(vec_true / vec_compare)
        z = chainer.Variable(np.zeros((len(nd), 1), dtype=np.float32))
        return F.sum(F.where(nd_var, include_nan, z))

    def jsd(self, vec_true, vec_compare):
        # Jensen–Shannon divergence
        vec_mean = 0.5 * (vec_true + vec_compare)
        return 0.5 * self.kld(vec_true, vec_mean) + 0.5 * self.kld(vec_compare, vec_mean)

    def topkprob(self, vec, k=5):
        vec_sort = np.sort(vec)[-1::-1]
        topk = vec_sort[:k]
        ary = np.arange(k)
        return np.prod([np.exp(topk[i]) / np.sum(np.exp(topk[i:])) for i in ary])

    def listwise_cost(self, list_ans, list_pred):
        return - np.sum(self.topkprob(list_ans) * np.log(self.topkprob(list_pred)))


class ListNet(NNfuncs.NN):
    def __init__(self, resumemodelName=None):
        self.resumemodelName = resumemodelName
        self.train_loss, self.test_loss = [], []
        self.train_acc, self.test_acc = [], []
        if resumemodelName is not None:
            print("load resume model!")
            self.loadModel(resumemodelName)

    def ndcg(self, y_true, y_score, k=20):
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        y_true_sorted = sorted(y_true, reverse=True)
        ideal_dcg = 0
        for i in range(k):
            ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
        dcg = 0
        argsort_indices = np.argsort(y_score)[::-1]
        for i in range(k):
            dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
        ndcg = dcg / ideal_dcg
        return ndcg

    def trainModel(self, x_train, y_train, x_test, y_test, n_epoch, batchsize):
        print("Start training and validation loop......")
        N = len(x_train)
        N_test = len(x_test)
        for epoch in range(1, n_epoch + 1):
            print('epoch', epoch)           
            # training
            perm = np.random.permutation(N)
            sum_loss = 0
            for i in tqdm(range(0, N, batchsize)):
                x = chainer.Variable(np.asarray(x_train[perm[i:i + batchsize]]))
                t = chainer.Variable(np.asarray(y_train[perm[i:i + batchsize]]))
                self.optimizer.update(self.model, x, t)
                sum_loss += float(self.model.loss.data) * len(t.data)

            print('train mean loss={}'.format(sum_loss / N))
            self.train_loss.append(sum_loss / N)

            perm = np.random.permutation(N_test)
            sum_loss = 0
            for j in tqdm(range(0, N_test, batchsize)):
                x = chainer.Variable(np.asarray(x_test[perm[j:j + batchsize]]), volatile='off')
                t = chainer.Variable(np.asarray(y_test[perm[j:j + batchsize]]), volatile='off')
                loss = self.model(x, t)
                sum_loss += float(loss.data) * len(t.data)
            print('test  mean loss={}'.format(sum_loss / N_test))
            self.test_loss.append(sum_loss / N_test)

            train_score = self.model.predict(chainer.Variable(x_train))
            test_score = self.model.predict(chainer.Variable(x_test))
            train_ndcg = self.ndcg(y_train, train_score)
            test_ndcg = self.ndcg(y_test, test_score)
            self.train_acc.append(train_ndcg)
            self.test_acc.append(test_ndcg)
            print("epoch: {0}".format(epoch + 1))
            print("NDCG@20 | train: {0}, test: {1}".format(train_ndcg, test_ndcg))

    def fit(self, fit_X, fit_y, batchsize=20, n_epoch=20, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam"):
        train_X, train_y, validate_X, validate_y = self.splitData(fit_X, fit_y, tv_ratio)
        print("The number of data, train:", len(train_X), "validate:", len(validate_X))
        if self.resumemodelName is None:
            self.initializeModel(Model, train_X, n_units1, n_units2, optimizerAlgorithm)
        self.trainModel(train_X, train_y, validate_X, validate_y, n_epoch, batchsize)

    def test(self, fit_X, fit_y, batchsize=10, n_epoch=1, tv_ratio=0.95, optimizerAlgorithm="Adam"):
        train_X, train_y, validate_X, validate_y = self.splitData(fit_X, fit_y, tv_ratio)
        print("The number of data, train:", len(train_X), "validate:", len(validate_X))
        self.trainModel(train_X, train_y, validate_X, validate_y, n_epoch, batchsize)

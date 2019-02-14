import numpy as np

class DataSet():
    def __init__(self, dataset):
        print('Preparing data set...')
        xtrain = np.load("{}nltk_xtrain.npy".format(dataset))
        ytrain = np.load("{}nltk_ytrain.npy".format(dataset))
        # ultrain 在'v_adv'中使用
        ultrain = np.load("{}nltk_ultrain.npy".format(dataset)) if (loss_type == 'v_adv') else None

        # defining validation set
        xval = list()
        yval = list()
        for _ in range(int(len(ytrain) * 0.025)):
            xval.append(xtrain[0])
            xval.append(xtrain[-1])
            yval.append(ytrain[0])
            yval.append(ytrain[-1])
            xtrain = np.delete(xtrain, 0)  # 删掉首尾，属于预处理
            xtrain = np.delete(xtrain, -1)
            ytrain = np.delete(ytrain, 0)
            ytrain = np.delete(ytrain, -1)
        xval = np.asarray(xval)
        yval = np.asarray(yval)
        print('{} elements in validation set'.format(len(yval)))
        # ---
        yval = np.reshape(yval, newshape=(yval.shape[0], 1))  # 去除冗余的维度
        ytrain = np.reshape(ytrain, newshape=(ytrain.shape[0], 1))

        # self.x =
        # self.y =
        # self.ul =

    def get_minibatch(self, batch_shape=(64,400)):
        x = self.pad_sequences(x, maxlen=batch_shape[1])
        permutations = np.random.permutation(len(y))
        ul_permutations = None
        len_ratio = None
        if (ul is not None):
            ul = K.preprocessing.sequence.pad_sequences(ul, maxlen=batch_shape[1])
            ul_permutations = np.random.permutation(len(ul))
            len_ratio = len(ul) / len(y)
        for s in range(0, len(y), batch_shape[0]):
            perm = permutations[s:s + batch_shape[0]]
            minibatch = {'x': x[perm], 'y': y[perm]}
            if (ul is not None):
                ul_perm = ul_permutations[int(np.floor(len_ratio * s)):int(np.floor(len_ratio * (s + batch_shape[0])))]
                minibatch.update({'ul': np.concatenate((ul[ul_perm], x[perm]), axis=0)})
            yield minibatch

    def pad_sequences(self):
        pass
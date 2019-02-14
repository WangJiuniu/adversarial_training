import torch
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
from atNet import AtNet
import numpy as np
from progressbar import ProgressBar
USE_CUDA = torch.cuda.is_available()

class AtModel(object):
    def __init__(self, embedding_weights, dropout=0.2, lstm_units=1024, dense_units=30):
        self.atNet = AtNet(embedding_weights, dropout=dropout, lstm_units=lstm_units, dense_units=dense_units)
        if USE_CUDA:
            self.atNet.cuda()
        parameters = [p for p in self.atNet.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters)

    def train(self, dataset, batch_shape, epochs, loss_type='none', p_mult=0.02):
        best_dev_acc = 0
        print('Training...')
        self.loss_type = loss_type
        xtrain = np.load("{}nltk_xtrain.npy".format(dataset))
        ytrain = np.load("{}nltk_ytrain.npy".format(dataset))

        # defining validation set
        xval = list()
        yval = list()
        for _ in range(int(len(ytrain) * 0.025)):
            # take some data into val from train
            xval.append(xtrain[0])
            xval.append(xtrain[-1])
            yval.append(ytrain[0])
            yval.append(ytrain[-1])
            xtrain = np.delete(xtrain, 0)
            xtrain = np.delete(xtrain, -1)
            ytrain = np.delete(ytrain, 0)
            ytrain = np.delete(ytrain, -1)
        xval = np.asarray(xval)
        yval = np.asarray(yval)
        print('{} elements in validation set'.format(len(yval)))

        _losses = list()
        _accuracies = list()
        for epoch in range(epochs):
            losses = list()
            accuracies = list()

            bar = ProgressBar(maxval=np.floor(len(ytrain) / batch_shape[0]).astype('i'))
            bar.start()
            minibatch = enumerate(self.get_minibatch(xtrain, ytrain, batch_shape=batch_shape))
            for i, train_batch in minibatch:
                fd = {'batch': train_batch['x'], 'labels': train_batch['y']}  # training mode
                acc_val, loss_val = self.update(fd, p_mult)
                accuracies.append(acc_val)
                losses.append(loss_val)
                bar.update(i)

            # saving accuracies and losses
            _accuracies.append(accuracies)
            _losses.append(losses)
            log_msg = "\nEpoch {} of {} -- average accuracy is {:.3f} (train) -- average loss is {:.3f}"
            print(log_msg.format(epoch + 1, epochs, np.asarray(accuracies).mean(), np.asarray(losses).mean()))
            # validation log
            dev_acc = self.validation(xval, yval, batch_shape=batch_shape, set_name='dev')
            if best_dev_acc < dev_acc:
                print('Get the best model, save it.')
                best_dev_acc = dev_acc
                torch.save(self.atNet.state_dict(), 'best_model_params.pkl')
            print('Best dev acc till now is {:.3f}'.format(best_dev_acc))
        print('\nTrain Finished!\n')

    def test(self, dataset, batch_shape=(64, 400)):
        print('Load the best model params and test...')
        self.atNet.load_state_dict(torch.load('best_model_params.pkl'))
        xtest = np.load("{}nltk_xtest.npy".format(dataset))
        ytest = np.load("{}nltk_ytest.npy".format(dataset))
        self.validation(xtest, ytest, batch_shape=batch_shape, set_name='test')

    def validation(self, x, y, batch_shape, set_name='dev'):
        print('{} validation...'.format(set_name))
        self.atNet.eval()  # test mode
        accuracies = list()
        minibatch = self.get_minibatch(x, y, batch_shape=batch_shape)
        for val_batch in minibatch:
            fd = {'batch': val_batch['x'], 'labels': val_batch['y']}
            if USE_CUDA:
                batch = Variable(torch.LongTensor(fd['batch']).cuda(async=True))
                labels = Variable(torch.LongTensor(fd['labels']).cuda(async=True))
            else:
                batch = Variable(torch.LongTensor(fd['batch']))
                labels = Variable(torch.LongTensor(fd['labels']))
            pred, _ = self.atNet(batch)
            acc_val = self.get_acc(labels, pred)
            accuracies.append(acc_val)
        dev_acc = np.asarray(accuracies).mean()
        print("Average accuracy on {} is {:.3f}".format(set_name, dev_acc))
        return dev_acc

    def update(self, fd, p_mult):
        if USE_CUDA:
            batch = Variable(torch.LongTensor(fd['batch']).cuda(async=True))
            labels = Variable(torch.LongTensor(fd['labels']).cuda(async=True))
        else:
            batch = Variable(torch.LongTensor(fd['batch']))
            labels = Variable(torch.LongTensor(fd['labels']))
        # ul_batch = fd['ul_batch'] if 'ul_batch' in fd else None
        self.atNet.train()
        pred, emb = self.atNet(batch)
        acc, loss = self._caculate_loss(batch, labels, pred, emb, p_mult)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.atNet.parameters(), 10)
        self.optimizer.step()
        return acc, loss.data.cpu().numpy()

    def _caculate_loss(self, batch, labels, pred, emb, p_mult):
        loss = F.cross_entropy(pred, labels)
        if (self.loss_type == 'adv'):
            emb_grad = grad(loss, emb, retain_graph=True)
            p_adv = torch.FloatTensor(p_mult * _l2_normalize(emb_grad[0].data))
            if USE_CUDA:
                p_adv = p_adv.cuda(async=True)
            p_adv = Variable(p_adv)
            adv_loss = F.cross_entropy(self.atNet(batch, p_adv)[0], labels)
            loss += adv_loss
        accuracy = self.get_acc(labels, pred)
        return accuracy, loss

    def get_acc(self, labels, pred):
        len = labels.size(0)
        labels = labels.data.cpu().numpy()
        _, predict = torch.max(pred, 1)
        predict = predict.data.cpu().numpy()
        num_correct = (labels == predict).sum()
        acc = num_correct * 1.0 / len
        return acc

    def get_minibatch(self, x, y, batch_shape):
        x = self.pad_sequences(x, maxlen=batch_shape[1])
        permutations = np.random.permutation(len(y))
        for s in range(0, len(y), batch_shape[0]):
            perm = permutations[s:s + batch_shape[0]]
            minibatch = {'x': x[perm], 'y': y[perm]}
            yield minibatch

    def pad_sequences(self, x, maxlen):
        pad_id = 0
        x = [(ids + [pad_id] * (maxlen - len(ids)))[:maxlen] for ids in x]
        return np.asarray(x)


def _l2_normalize(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


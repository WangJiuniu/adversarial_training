import sys
sys.path.append('..')
import os
import numpy as np
from model import AtModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(data, n_epochs, n_ex, ex_len, lt, pm):
    """
    Train and test using IMDB data (can choose whether use adversarial training).
    :param data: path of the folder that contains the data (train, test, emb. matrix)
    :param n_epochs: number of training epochs
    :param n_ex: Number of EXamples per batch
    :param ex_len: Lenght of each Example
    :param lt: choose whether use adversarial training ('none' or 'adv')
    :param pm: Perturbation Multiplier (used with adv)
    """
    embedding_weights = np.load("{}nltk_embedding_matrix.npy".format(data))
    atModel = AtModel(embedding_weights)
    atModel.train(data, batch_shape=(n_ex, ex_len), epochs=n_epochs, loss_type=lt, p_mult=pm)
    atModel.test(data, batch_shape=(n_ex, ex_len))

print("----------  lt='none' ----------")
main(data='../imdb/', n_epochs=10, n_ex=128, ex_len=400, lt='none', pm=0.02)
# main(data='../imdb/', n_epochs=3, n_ex=128, ex_len=40, lt='none', pm=0.02)

print("----------  lt='adv' ----------")
main(data='../imdb/', n_epochs=10, n_ex=128, ex_len=400, lt='adv', pm=0.02)
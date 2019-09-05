from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os

GLOVE_50_PATH = "Datasets/StandardGlove/glove.6B.50d.txt"
GLOVE_100_PATH = "Datasets/StandardGlove/glove.6B.100d.txt"
GLOVE_200_PATH = "Datasets/StandardGlove/glove.6B.200d.txt"
GLOVE_300_PATH = "Datasets/StandardGlove/glove.6B.300d.txt"


def _loadStandardGloveVectors(path):
    fileNameStart = os.path.realpath(__file__).rfind('/')
    filePath = os.path.realpath(__file__)[:fileNameStart] + "/" + path  # Remove File Name from path
    return loadGloveVectors(filePath)


def loadGloveVectors(filePath):
    glove_file = datapath(filePath)
    tmp_file = get_tmpfile("test_word2vec.txt")  # Temp file, should probably be removed later
    print("Loading Glove from:", filePath)
    glove2word2vec(glove_file, tmp_file)
    print("Glove Loaded")
    return KeyedVectors.load_word2vec_format(tmp_file)


def saveKeyedVectorsAsWord2VecText(model, path):
    model.save_word2vec_format(path)


def loadWord2Vec(filePath):
    return KeyedVectors.load_word2vec_format(filePath)


def loadKeyedVectors(filePath):
    return KeyedVectors.load(filePath, mmap='r')


def loadGlove50():
    return _loadStandardGloveVectors(GLOVE_50_PATH)


def loadGlove100():
    return _loadStandardGloveVectors(GLOVE_100_PATH)


def loadGlove200():
    return _loadStandardGloveVectors(GLOVE_200_PATH)


def loadGlove300():
    return _loadStandardGloveVectors(GLOVE_300_PATH)

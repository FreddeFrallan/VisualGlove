from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os

GLOVE_50_PATH = "Datasets/StandardGlove/glove.6B.50d.txt"
GLOVE_100_PATH = "Datasets/StandardGlove/glove.6B.100d.txt"
GLOVE_200_PATH = "Datasets/StandardGlove/glove.6B.200d.txt"
GLOVE_300_PATH = "Datasets/StandardGlove/glove.6B.300d.txt"


def _loadGloveVectors(filePath):
    fileNameStart = os.path.realpath(__file__).rfind('/')
    filePath = os.path.realpath(__file__)[:fileNameStart] + "/" + filePath # Remove File Name from path

    glove_file = datapath(filePath)
    tmp_file = get_tmpfile("test_word2vec.txt") # Temp file, should probably be removed later
    glove2word2vec(glove_file, tmp_file)
    return KeyedVectors.load_word2vec_format(tmp_file)


def loadGlove50():
    return _loadGloveVectors(GLOVE_50_PATH)


def loadGlove100():
    return _loadGloveVectors(GLOVE_100_PATH)


def loadGlove200():
    return _loadGloveVectors(GLOVE_200_PATH)


def loadGlove300():
    return _loadGloveVectors(GLOVE_300_PATH)

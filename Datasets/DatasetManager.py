from Utils import ImageProcessing
from PIL import Image
import numpy as np
import os

DATASET_FOLDER_NAME = "Datasets"
STANDARD_GLOVE_FOLDER_NAME = "StandardGlove"
GLOVE_VOCAB_CSV_NAME = "GloveVocab.csv"
BING_IMAGES_ABSOLUTE_PATH = "/home/ubuntu/GloveVocab/Images"

VISUAL_EMBEDDINGS_FOLDER_NAME = "VisualEmbeddings"
FULL_SIZE_FOLDER_NAME = "FullSize"

BLESS_FILE_PATH = "BLESS/bless.txt"
SYNONYMS_FILE_PATH = "Synonyms/Synonyms and Antonyms - Samuel Fallows.txt"


################## File Managing

def _addFolderToPath(main, add):
    if (main[-1] == '/'):
        return "{}{}/".format(main, add)
    else:
        return "{}/{}/".format(main, add)


def _addFileToPath(main, add):
    if (main[-1] == '/'):
        return "{}{}".format(main, add)
    else:
        return "{}/{}".format(main, add)


def _getDatasetsFolderPath():
    return os.path.dirname(os.path.realpath(__file__))


################### Visual Glove
def getVisualGloveFolder(word):
    wordPath = _addFolderToPath(BING_IMAGES_ABSOLUTE_PATH, word)
    if (os.path.isdir(wordPath)):
        return wordPath
    raise Exception("{} does not exist".format(wordPath))


def getGloveImagesPaths(word):
    return [path for path in os.listdir(getVisualGloveFolder(word))]


def getGloveImages(word, asNumpy=True):
    try:
        folder = getVisualGloveFolder(word)  # Perhaps we don't have this word?
    except Exception as e:
        return []

    imgs = []
    for f in os.listdir(folder):
        try:
            img = Image.open(_addFileToPath(folder, f))

            if (img.mode == 'RGBA'):  # Removes Alpha channel from PNG
                img = ImageProcessing.removeAlphaChannel(img)
            if (img.mode != 'RGB'):
                raise Exception("Invalid image type: {}".format(img.mode))

            if (asNumpy):
                img = np.asarray(img)

            imgs.append(img)
        except Exception as e:
            # print(e)
            pass

    return imgs


def getSameSizeGloveImages(word, size, asNumpy=True):
    imgs = ImageProcessing.resizeImages(getGloveImages(word, asNumpy=False), size)
    return ImageProcessing.images2Numpy(imgs) if asNumpy else imgs


def getGloveBlendImage(word, size, asNumpy=True):
    imgs = getSameSizeGloveImages(word, size, asNumpy=True)
    return ImageProcessing.blendImages(imgs, asNumpy)


def getVisualEmbeddingsFolderPath(version=1):
    if (version == 1):
        return _addFolderToPath(_getDatasetsFolderPath(), VISUAL_EMBEDDINGS_FOLDER_NAME)
    else:
        return _addFolderToPath(_addFolderToPath(_getDatasetsFolderPath(), "V{}".format(version)),
                                VISUAL_EMBEDDINGS_FOLDER_NAME)


def getVisualEmbeddingsFullSizeFolderPath(version=1):
    return _addFolderToPath(getVisualEmbeddingsFolderPath(version), FULL_SIZE_FOLDER_NAME)


############################# Normal Glove

def getNormalGloveFolderPath():
    return _addFolderToPath(_getDatasetsFolderPath(), STANDARD_GLOVE_FOLDER_NAME)


def getGloveVocabCSVPath():
    return _addFileToPath(getNormalGloveFolderPath(), GLOVE_VOCAB_CSV_NAME)


############################ Embedding Manager
def getWordsAndEmbeddingsFromFile(filePath, asStr=False):
    counter = 0
    collectedWords = {}
    with open(filePath, 'r', encoding='utf-16') as file:
        for line in file:
            words = [w.strip() for w in line.split(' ') if len(w) > 0 and str.isspace(w) == False]  # Remove Whitespace
            if (asStr == False):
                collectedWords[words[0]] = np.array([float(w) for w in words[1:]])
            else:
                collectedWords[words[0]] = words[1:]
            if (counter % 1000 == 0):
                print(counter, words[0])
            counter += 1

    return collectedWords


def _getWordsFromEmbeddingFile(filePath):
    collectedWords = []
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split(' ')
            collectedWords.append(words[0])
    return collectedWords


############################ Benchmarks Manager

def getBLESSDataset():
    return _addFileToPath(_getDatasetsFolderPath(), BLESS_FILE_PATH)


def getSynonymsDataset():
    return _addFileToPath(_getDatasetsFolderPath(), SYNONYMS_FILE_PATH)

from Utils import ImageProcessing
from PIL import Image
import numpy as np
import os

DATASET_FOLDER_NAME = "Datasets"
STANDARD_GLOVE_FOLDER_NAME = "StandardGlove"
GLOVE_VOCAB_CSV_NAME = "VisualGloveVocab.csv"
GLOVE_ABSOLUTE_PATH = "/media/fredrik/CE3B-25E9/unpacked/Images"

VISUAL_EMBEDDINGS_FOLDER_NAME = "VisualEmbeddings"
FULL_SIZE_FOLDER_NAME = "FullSize"


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
    gloveFolder = GLOVE_ABSOLUTE_PATH
    wordPath = _addFolderToPath(gloveFolder, word)
    if (os.path.isdir(wordPath)):
        return _addFolderToPath(gloveFolder, word)
    raise Exception()


def getGloveImagesPaths(word):
    return [path for path in os.listdir(getVisualGloveFolder(word))]


def getGloveImages(word, asNumpy=True):
    try:
        folder = getVisualGloveFolder(word)  # Perhaps we don't have this word?
    except:
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


def getVisualEmbeddingsFullSizeFolderPath():
    return _addFolderToPath(_addFolderToPath(_getDatasetsFolderPath(), VISUAL_EMBEDDINGS_FOLDER_NAME),
                            FULL_SIZE_FOLDER_NAME)


############################# Normal Glove

def getGloveFolderPath():
    return _addFolderToPath(_getDatasetsFolderPath(), STANDARD_GLOVE_FOLDER_NAME)


def getGloveVocabCSVPath():
    return _addFileToPath(getGloveFolderPath(), GLOVE_VOCAB_CSV_NAME)


############################ Embedding Manager
def getWordsAndEmbeddingsFromFile(filePath, asStr=False):
    counter = 0
    collectedWords = {}
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
            words = [w.strip() for w in line.split(' ') if len(w) > 0 and str.isspace(w) == False]  # Remove Whitespace
            if(asStr == False):
                collectedWords[words[0]] = np.array([float(w) for w in words[1:]])
            else:
                collectedWords[words[0]] = words[1:]
            if(counter % 1000 == 0):
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

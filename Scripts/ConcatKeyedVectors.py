from NormalGlove import GloveFormatter
from Datasets import DatasetManager
import os

STANDARD_COMBINATIONS = [(50, 50), (50, 150), (100, 100), (100, 200), (200, 100)]


def _getCombinedGloveFolder(name, gloveSize, visualSize, version=1):
    if (version == 1):
        return DatasetManager._getDatasetsFolderPath() + "/CombinedGlove-{}/Glove{}-Visual{}/".format(name, gloveSize,
                                                                                                      visualSize)
    else:
        return DatasetManager._getDatasetsFolderPath() + "/V{}/CombinedGlove-{}/Glove{}-Visual{}/".format(version, name,
                                                                                                          gloveSize,
                                                                                                          visualSize)


def _getCombinedGlovedFilename(name, gloveSize, visualSize, version=1):
    return _getCombinedGloveFolder(name, gloveSize, visualSize, version) + "CombinedGlove-{}-{}.txt".format(gloveSize,
                                                                                                            visualSize)


def _getCombinedKeyedFilename(name, gloveSize, visualSize, version=1):
    return _getCombinedGloveFolder(name, gloveSize, visualSize, version) + "Keyed-Glove{}-Visual{}".format(gloveSize,
                                                                                                           visualSize)


def concatToStandardGlove(visualGloveBaseFolder, name, sizeCombinations=None, version=1):
    if (sizeCombinations == None):
        sizeCombinations = STANDARD_COMBINATIONS

    for gloveSize, visualGloveSize in sizeCombinations:
        gloveFile = DatasetManager._getDatasetsFolderPath() + "/StandardGlove/Keyed-Glove{}-Visual0".format(gloveSize)
        visualGloveFile = visualGloveBaseFolder + "-{}/Keyed-VisualGlove-{}".format(visualGloveSize, visualGloveSize)

        saveDir = _getCombinedGloveFolder(name, gloveSize, visualGloveSize, version)
        if (os.path.isdir(saveDir) == False):
            os.makedirs(saveDir)

        newFileName = _getCombinedGlovedFilename(name, gloveSize, visualGloveSize, version)
        GloveFormatter.combineGloveFiles(visualGloveFile, gloveFile, newFileName)

        keyedFilename = _getCombinedKeyedFilename(name, gloveSize, visualGloveSize, version)
        GloveFormatter.createKeyedVectorsFromGloveFile(newFileName, keyedFilename)


def main():
    datasetPath = DatasetManager._getDatasetsFolderPath()
    print(datasetPath)
    sizes = [(50, 50), (50, 150), (100, 100), (100, 200), (200, 100)]
    for gloveSize, visualGloveSize in sizes:
        gloveFile = datasetPath + "/StandardGlove/glove.6B.{}d.txt".format(gloveSize)
        visualGloveFile = datasetPath + "/VisualEmbeddings/Top-100K-{}/VisualGlove-{}.txt".format(
            visualGloveSize, visualGloveSize)

        saveDir = "/home/ubuntu/VisualGlove/Datasets/CombinedGlove-Top100K/Glove{}-Visual{}/".format(gloveSize,
                                                                                                     visualGloveSize)
        if (os.path.isdir(saveDir) == False):
            os.makedirs(saveDir)
        newFileName = saveDir + "CombinedGlove-{}-{}.txt".format(gloveSize, visualGloveSize)
        GloveFormatter.combineGloveFiles(visualGloveFile, gloveFile, newFileName)

        keyedFilename = saveDir + "Keyed-Glove{}-Visual{}".format(gloveSize, visualGloveSize)
        GloveFormatter.createKeyedVectorsFromGloveFile(newFileName, keyedFilename)

from Datasets import DatasetManager
from NormalGlove import GloveFormatter
import os


def main():
    glovePath = DatasetManager.getVisualEmbeddingsFolderPath(2) + "/VisualGlove-2.0 Full.txt"
    GloveFormatter.createKeyedVectorsFromGloveFile(glovePath, "Keyed-VisualGlove-2.0-Full")
    '''

    skipSize = 0
    basePath = DatasetManager.getVisualEmbeddingsFolderPath(2)
    baseName = "Top-400K-Skip0"
    for d in [25, 50, 100, 150, 200, 300]:
        folderPath = basePath + "/{}-{}/".format(baseName, d)
        if (os.path.isdir(folderPath) == False):
            os.mkdir(folderPath)

        glovePath = folderPath + "VisualGlove-{}{}.txt".format(baseName, d)
        keyPath = folderPath + "Keyed-VisualGlove-{}".format(d)
        GloveFormatter.createKeyedVectorsFromGloveFile(glovePath, keyPath)
    '''


if (__name__ == '__main__'):
    main()

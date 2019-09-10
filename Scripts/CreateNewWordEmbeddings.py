from NormalGlove import GloveFormatter
from Datasets import DatasetManager
from Scripts import PerformPCA, ConcatKeyedVectors


def main():
    version = 2
    skipSize = 0
    includeSize = 400000
    dimensions = [25, 50, 100, 150, 200, 300]

    name = "Top{}K-Skip{}".format(round(includeSize / 1000), skipSize)
    gloveOutputFolder = DatasetManager.getVisualEmbeddingsFolderPath(version) + name

    '''
    PerformPCA.performPCA(name, gloveOutputFolder, dimensions, includeSize, skipSize, version)

    # Convert into KeyedVectors
    for d in dimensions:
        folderPath = gloveOutputFolder + "-{}/".format(d)
        glovePath = folderPath + "VisualGlove-{}-{}.txt".format(name, d)
        keyPath = folderPath + "Keyed-VisualGlove-{}".format(d)
        GloveFormatter.createKeyedVectorsFromGloveFile(glovePath, keyPath)

    '''
    # Concat with standard Glove
    sizes = [(100, 300), (50, 300)]
    ConcatKeyedVectors.concatToStandardGlove(gloveOutputFolder, name, sizeCombinations=sizes, version=version)

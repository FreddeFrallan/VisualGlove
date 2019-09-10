from Datasets import DatasetManager
from Visuals import PCAReduction
from NormalGlove import Model
import os


def performPCA(mainFilename, gloveOutputFolder, dimensions, includeSize, skipSize=0, version=1):
    skipDimensions = [d + skipSize for d in dimensions]

    PCAOutputFolder = DatasetManager.getVisualEmbeddingsFullSizeFolderPath(version) + "PCA-{}".format(mainFilename)
    if (os.path.isdir(PCAOutputFolder) == False):
        os.mkdir(PCAOutputFolder)

    embeddingFilePath = DatasetManager.getVisualEmbeddingsFullSizeFolderPath(version) + "/Keyed-VisualGlove-Full"
    model = Model.loadKeyedVectors(embeddingFilePath)
    pureEmbeddings = [model.wv[k] for k in model.vocab]

    PCAReduction.createPCATransformers(pureEmbeddings[:includeSize], skipDimensions, PCAOutputFolder)
    PCAReduction.createPCAEmbeddingFiles(model, pureEmbeddings, PCAOutputFolder, gloveOutputFolder, mainFilename,
                                         skipDimensions, skipSize)

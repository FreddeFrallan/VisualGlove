from NormalGlove import GloveFormatter
from sklearn.decomposition import PCA
from Utils import FileProcessing
import pickle, os


def _getPCATransformerFileName(dimensions):
    return 'VisualGlove-Full-PCA-Transformer-{}'.format(dimensions)


def _getSubGloveOutputFolder(base, dim):
    return base + "-{}".format(dim)


def _getFullSubGloveOutputPath(base, name, dim):
    return _getSubGloveOutputFolder(base, dim) + "/VisualGlove-{}-{}.txt".format(name, dim)


def createPCAEmbeddingFiles(model, pureEmbeddings, PCAFolderPath, outputFolder, name, dimensions, skipDimensions=0):
    for d in dimensions:
        realDim = d - skipDimensions
        print(_getPCATransformerFileName(d))
        with open(PCAFolderPath + "/" + _getPCATransformerFileName(d), 'rb') as file:
            pca = pickle.load(file)
            reducedEmbeddings = {}
            print("Performing PCA for:", d)
            pcaEmbeddings = pca.transform(pureEmbeddings)

            print("Creating data structure")
            for i, w in enumerate(model.vocab):
                reducedEmbeddings[w] = pcaEmbeddings[i][skipDimensions:]

            fullOutFolder = _getSubGloveOutputFolder(outputFolder, realDim)
            if (os.path.isdir(fullOutFolder) == False):
                os.mkdir(fullOutFolder)
            print("Storing to disk in:", fullOutFolder)

            fullOutputPath = _getFullSubGloveOutputPath(outputFolder, name, realDim)
            GloveFormatter.createGloveFile(reducedEmbeddings, fullOutputPath)


def createPCATransformer(embeddings, newSize):
    pca = PCA(n_components=newSize)
    pca.fit_transform(embeddings)
    return pca


def createPCATransformers(pureEmbeddings, dimensions, outputFolder=""):
    if (outputFolder != "" and outputFolder[-1] != '/'):
        outputFolder += "/"

    for d in dimensions:
        print("Performing PCA on:", d)
        pca = createPCATransformer(pureEmbeddings, d)
        print("Saving PCA")
        FileProcessing.saveToFile(pca, outputFolder + _getPCATransformerFileName(d))

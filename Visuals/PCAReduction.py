from sklearn.decomposition import PCA
from Datasets import DatasetManager
from NormalGlove import GloveFormatter
from Utils import FileProcessing
import pickle


def _getPCATransformerFileName(dimensions):
    return 'VisualGlove-Full-PCA-Transformer-{}'.format(dimensions)


def _getPCAEmbeddingsFileName(dimensions):
    return 'PCAEmbeddings{}'.format(dimensions)


def createPCAEmbeddingFiles(embeddings, PCAFolderPath, dimensions):
    pureEmbeddings = [embeddings[k] for k in embeddings.keys()]

    for d in dimensions:
        print(_getPCATransformerFileName(d))
        with open(PCAFolderPath + "/" + _getPCATransformerFileName(d), 'rb') as file:
            pca = pickle.load(file)
            reducedEmbeddings = {}
            print("Performing PCA for:", d)
            pcaEmbeddings = pca.transform(pureEmbeddings)

            print("Creating data structure")
            for i, k in enumerate(embeddings.keys()):
                reducedEmbeddings[k] = pcaEmbeddings[i]

            print("Storing to disk")
            GloveFormatter.createGloveFile(reducedEmbeddings, "VisualGlove-{}.txt".format(d))


def createPCATransformer(embeddings, newSize):
    pca = PCA(n_components=newSize)
    pca.fit_transform(embeddings)
    return pca


def createPCATransformers(embeddings, dimensions=[50, 100, 150, 200, 300], outputFolder=""):
    pureEmbeddings = [embeddings[k] for k in embeddings.keys()]
    if (outputFolder != ""):
        outputFolder += "/"

    for d in dimensions:
        print("Performing PCA on:", d)
        pca = createPCATransformer(pureEmbeddings, d)
        print("Saving PCA")
        FileProcessing.saveToFile(pca, outputFolder + _getPCATransformerFileName(d))


def main():
    embeddingFilePath = DatasetManager.getVisualEmbeddingsFullSizeFolderPath() + "/VisualGlove-Full.txt"
    embeddings = DatasetManager.getWordsAndEmbeddingsFromFile(embeddingFilePath)
    outputFolder = DatasetManager.getVisualEmbeddingsFullSizeFolderPath() + "/PCA"
    #createPCATransformers(embeddings, [25], outputFolder)

    pcaFolderPath = DatasetManager.getVisualEmbeddingsFullSizeFolderPath() + "/PCA"
    createPCAEmbeddingFiles(embeddings, pcaFolderPath, [25, 50, 100, 150, 200, 300])


if (__name__ == '__main__'):
    main()

from sklearn.manifold import TSNE
from Utils import FileProcessing
import matplotlib.pyplot as plt


def visualizeEmbeddings(embeddings, labels):
    x, y = [], []
    for value in embeddings:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def getTsneEmbeddings(embeddings, perplexity=40, iterations=2500):
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=iterations, random_state=42, verbose=1)
    return tsne_model.fit_transform(embeddings)


def main():
    from Datasets import DatasetManager

    embeddingPath = DatasetManager.getVisualEmbeddingsFullSizeFolderPath() + "VisualGlove-Full.txt"
    embeddings = DatasetManager.getWordsAndEmbeddingsFromFile(embeddingPath)
    FileProcessing.saveToFile(embeddings, 'tempfullEmbeddings')

    pureEmbeddings = [embeddings[w] for w in embeddings.keys()]
    print("Getting TSNE embeddings")
    reducedEmbeddings = getTsneEmbeddings(pureEmbeddings)

    print("Saving to file...")
    FileProcessing.saveToFile(reducedEmbeddings, 'tempEmbeddings')

    labels = [k for k in embeddings.keys()]
    visualizeEmbeddings(reducedEmbeddings, labels)


if (__name__ == '__main__'):
    main()

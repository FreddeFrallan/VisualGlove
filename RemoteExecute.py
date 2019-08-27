from Visuals.ImageEmbedder import createAverageEmbeddingsForVocab
from Datasets import DatasetManager

if __name__ == '__main__':
    from NormalGlove import Vocab
    path = DatasetManager.getVisualEmbeddingsFullSizeFolderPath()
    vocab = Vocab.readVocabFromCSVFile()
    createAverageEmbeddingsForVocab(vocab, 3, 10, [224, 224], path, 3)
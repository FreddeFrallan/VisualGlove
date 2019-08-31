from Visuals import ImageEmbedder, Leftovers
from Datasets import DatasetManager
import sys

def findLeftovers():
    from NormalGlove import Vocab
    path = DatasetManager.getVisualEmbeddingsFullSizeFolderPath()
    vocab = Vocab.readVocabFromCSVFile()
    lostEmbeddings = Leftovers.findWordsThatLackEmbedings(vocab, path)
    print("Leftovers found", len(lostEmbeddings))

if __name__ == '__main__':
    #createAverageEmbeddingsForVocab(vocab, 3, 10, [224, 224], path, 3)
    if(sys.argv[0].low() == "leftovers"):
        findLeftovers()
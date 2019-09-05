from Datasets import DatasetManager
import os


def findWordsThatLackEmbedings(fullVocab, folderPathToEmbeddingsFiles):
    wordsFoundInFolders = {}
    counter = 0
    for filePath in os.listdir(folderPathToEmbeddingsFiles):
        print("File", counter)
        for w in DatasetManager._getWordsFromEmbeddingFile(folderPathToEmbeddingsFiles + "/" + filePath):
            if (w not in wordsFoundInFolders):
                wordsFoundInFolders[w] = 1
        counter += 1

    wordsNotFound = []
    for w in fullVocab:
        if (w not in wordsFoundInFolders):
            wordsNotFound.append(w)

    return wordsNotFound



if (__name__ == '__main__'):
    from NormalGlove import Model

    # removeListCharsFromFile(DatasetManager.getVisualEmbeddingsFullSizeFolderPath() + "/VisualGlove-Full.txt", "NewFile")
    file = DatasetManager.getVisualEmbeddingsFullSizeFolderPath() + "/ProperFormat.txt"
    # currentVocab = DatasetManager._getWordsFromEmbeddingFile(file)
    # concatenateEmbeddingsFiles(DatasetManager.getVisualEmbeddingsFullSizeFolderPath(), "ProperFormat.txt")
    temp = Model.loadGloveVectors(file)

    '''
    fullVocab = Vocab.readVocabFromCSVFile()
    print("Full Vocab loaded")
    print("Current vocab loaded")

    currentVocabLookup = {}
    for w in currentVocab:
        currentVocabLookup[w] = 1
    del currentVocab

    fullVocabSize = len(fullVocab)
    missingVocab = []
    for i, w in enumerate(fullVocab):
        if(w not in currentVocabLookup):
            missingVocab.append(w)

    del fullVocab

    print("Missing vocab size:", len(missingVocab))
    createZeroVectors(missingVocab, 1280, "VisualGlove-Full-MissingWords.txt")

    '''

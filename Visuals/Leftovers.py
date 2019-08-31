def _getWordsFromEmbeddingFile(filePath):
    with open(filePath, 'r') as file:
        words = [line.split(' ')[0] for line in file.readlines()]
        return [w for w in words if len(w) > 0 and str.isspace(w) == False]


def findWordsThatLackEmbedings(fullVocab, folderPathToEmbeddingsFiles):
    import os

    wordsFoundInFolders = {}
    counter = 0
    for filePath in os.listdir(folderPathToEmbeddingsFiles):
        print("File", counter)
        for w in _getWordsFromEmbeddingFile(folderPathToEmbeddingsFiles + "/" + filePath):
            if(w not in wordsFoundInFolders):
                wordsFoundInFolders[w] = 1
        counter += 1

    wordsNotFound = []
    for w in fullVocab:
        if(w not in wordsFoundInFolders):
            wordsNotFound.append(w)

    return wordsNotFound


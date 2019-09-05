from Datasets import DatasetManager
from NormalGlove import Model
import numpy as np
import os


def _embeddingsToString(embeddings, strEmbeddings=False):
    if (strEmbeddings):
        return " ".join(embeddings)
    else:
        return " ".join(map(str, embeddings))


def createGloveFile(embeddings, fileName):
    with open(fileName, 'w', encoding='utf-8') as newFile:
        for i, w in enumerate(embeddings.keys()):
            newFile.write("{} {}\n".format(w, _embeddingsToString(embeddings[w])))
            if (i % 1000 == 0):
                print(i)


def concatenateEmbeddingsFiles(folderPath, newFileName):
    embeddedWords = {}
    with open(newFileName, 'w', encoding='utf-8') as newFile:
        for filePath in [p for p in os.listdir(folderPath) if os.path.isdir(folderPath + "/" + p) == False]:
            print(filePath)
            localEmbeddings = DatasetManager.getWordsAndEmbeddingsFromFile(folderPath + "/" + filePath, asStr=True)
            for i, w in enumerate(localEmbeddings):
                if (w not in embeddedWords):
                    embeddedWords[w] = 1
                    newFile.write("{} {}\n".format(w, _embeddingsToString(localEmbeddings[w], strEmbeddings=True)))
            print("Processed lines:", len(embeddedWords.keys()))


def createZeroVectors(vocab, vectorSize, fileName):
    with open(fileName, 'w', encoding='utf-8') as newFile:
        vector = _embeddingsToString(np.zeros(vectorSize))
        for i, w in enumerate(vocab):
            newFile.write("{} {}\n".format(w, vector))


def combineGloveFiles(file1, file2, outputFile):
    embeddings = Model.loadGloveVectors(file1).wv
    embeddings2 = Model.loadGloveVectors(file2).wv
    print("Vocab1:", len(embeddings.vocab), "  Vocab2:", len(embeddings2.vocab))
    vocab = list(embeddings.vocab)
    missingWords = []
    #embeddings = DatasetManager.getWordsAndEmbeddingsFromFile(file1, asStr=True)
    #embeddings2 = DatasetManager.getWordsAndEmbeddingsFromFile(file2, asStr=True)
    with open(outputFile, 'w', encoding='utf-8') as newFile:
        for i, w in enumerate(vocab):
            emb1 = _embeddingsToString(embeddings[w])
            if(w in embeddings2):
                emb2 = _embeddingsToString(embeddings2[w])
                newFile.write("{} {} {}\n".format(w, emb1, emb2))
                if (i % 10000 == 0):
                    print(i)
            else:
                missingWords.append(w)
        print("Missing words:", len(missingWords))
        print(missingWords)


def createKeyedVectorsFromGloveFile(gloveFile, outputFileName):
    print("Creating Keyed Vectors file from:", gloveFile)
    Model.loadGloveVectors(gloveFile).save(outputFileName)

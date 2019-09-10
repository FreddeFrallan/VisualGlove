from Datasets import DatasetManager
from NormalGlove import GloveFormatter
from Visuals import EmbeddingModel
from Utils import FileProcessing
import multiprocessing as mp
import numpy as np
import csv, os

BASE_FIND_FOLDER = "/home/ubuntu/GloveVocab/Images/"
BASE_STORE_RESIZED_FOLDER = "/home/ubuntu/ResizedImages/"


def _getResizedWordPath(word):
    return BASE_STORE_RESIZED_FOLDER + "Image-{}".format(word)


def _loadVocab():
    with open(DatasetManager.getGloveVocabCSVPath(), 'r') as f:
        vocab = list(csv.reader(f))[0]
    return vocab


def _resizeWorker(id, vocab, imgSize):
    print("Starting worker", id)
    failed = 0
    for i, w in enumerate(vocab):
        try:
            if ((i + 1) % 10 == 0):
                print("{}: {}/{}  Failed: {}".format(id, i, len(vocab), failed))

            imgs = DatasetManager.getSameSizeGloveImages(w, imgSize, asNumpy=True)
            if(len(imgs) == 0):
                failed += 1
                continue

            FileProcessing.saveToFile(imgs, _getResizedWordPath(w))
        except Exception as e:
            failed += 1
            pass


def resizeAllImages():
    numberOfWorkers = 20
    imgSize = (331, 331)

    vocab = _loadVocab()
    print("Vocab Size:", len(vocab))

    procs = []
    procVocabSize = np.ceil(len(vocab) / numberOfWorkers)
    for i in range(numberOfWorkers):
        pVocab = vocab[int(i * procVocabSize):int((i + 1) * procVocabSize)]
        procs.append(mp.Process(target=_resizeWorker, args=(i, pVocab, imgSize)))

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def _imageWorker(id, vocab, wordsPerBatch, sendPipe, listenPipe):
    print("Starting image worker:", id)
    vocabSize = len(vocab)
    counter = 0
    while (listenPipe.get() != 'QUIT' and counter < vocabSize):
        currentWords = vocab[counter:counter + wordsPerBatch]
        counter += wordsPerBatch

        words, imagesToProcess = [], []
        for w in currentWords:
            try:
                if (os.path.isfile(_getResizedWordPath(w)) == False):
                    print("No file for:", w)
                    continue
                imgs = FileProcessing.loadFromFile(_getResizedWordPath(w))
                print(imgs)
                print(len(imgs))
                if (len(imgs) > 0):
                    words.append(w)
                    imagesToProcess.append(imgs)
            except Exception as e:
                print(e)
                pass

        if (len(words) == 0):
            print(id, "Sent no IMAGES")
            continue

        sendPipe.put((id, words, imagesToProcess))
    print("Ending image worker:", id)


def createAverageEmbeddingsForVocab():
    wordsPerBatch = 20
    wordsPerFile = 10000
    numberOfThreads = 10

    vocab = _loadVocab()
    vocabSize = len(vocab)
    print("Vocab Size:", vocabSize)

    embeddingModel = EmbeddingModel.createImageEmbeddingModel()

    # Setup Image Procs
    procs = []
    toPipes = [mp.Queue() for i in range(numberOfThreads)]
    listenPipe = mp.Queue()
    wordsPerWorker = np.ceil(vocabSize / numberOfThreads)
    for i in range(numberOfThreads):
        procVocab = vocab[int(i * wordsPerWorker): int((i + 1) * wordsPerWorker)]
        procs.append(mp.Process(target=_imageWorker, args=(i, procVocab, wordsPerBatch, listenPipe, toPipes[i])))

    # Start all image procs
    for i, p in enumerate(procs):
        p.start()
        toPipes[i].put('Get Images')

    # Setup main pipeline
    filesCreated = 0
    totalEmbeddingsSaved = 0
    embeddingsInCurrentFile = 0
    # currentFile = open(DatasetManager._addFileToPath(outFolder, "LeftOverEmbeddings{}.txt".format(filesCreated)), 'w')
    storedEmbeddings = {}
    batchCounter = 0
    while (totalEmbeddingsSaved < vocabSize):
        id, words, imgs = listenPipe.get()
        if (len(words) == 0):
            continue

        try:
            for i, embedding in enumerate(EmbeddingModel.createAverageEmbeddingForImages(embeddingModel, imgs)):
                storedEmbeddings[words[i]] = embedding
                # currentFile.write("{} {}\n".format(words[i], np.array2string(embedding)[1:-1]))

            embeddingsInCurrentFile += wordsPerBatch
            totalEmbeddingsSaved += wordsPerBatch
            #if (batchCounter % 100 == 0):
            print(totalEmbeddingsSaved, "/", vocabSize)
            batchCounter += 1

            if (embeddingsInCurrentFile >= wordsPerFile):
                GloveFormatter.createGloveFile(storedEmbeddings, "NASNetLarg-Embeddings{}.txt".format(filesCreated))
                embeddingsInCurrentFile = 0
                storedEmbeddings = {}
                filesCreated += 1

        except Exception as e:
            print(e)

        toPipes[id].put("Get Images")

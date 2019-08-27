from Datasets import DatasetManager
from Visuals import EmbeddingModel
import numpy as np

def _imageWorker(id, vocab, wordsPerBatch, imgSize, sendPipe, listenPipe):
    print("Starting image worker:", id)
    from Datasets import DatasetManager
    vocabSize = len(vocab)
    counter = 0
    while (listenPipe.get() != 'QUIT' and counter < vocabSize):
        currentWords = vocab[counter:counter + wordsPerBatch]
        counter += wordsPerBatch

        words, imagesToProcess = [], []
        for w in currentWords:
            imgs = DatasetManager.getSameSizeGloveImages(w, imgSize, asNumpy=True)
            if (len(imgs) > 0):
                words.append(w)
                imagesToProcess.append(imgs)

        sendPipe.put((id, words, imagesToProcess))
    print("Ending image worker:", id)


def createAverageEmbeddingsForVocab(vocab, wordsPerBatch, wordsPerFile, imgSize, outFolder, numberOfThreads):
    import multiprocessing as mp
    vocabSize = len(vocab)

    np.set_printoptions(suppress=True, linewidth=999999, threshold=9999999)
    embeddingModel = EmbeddingModel.createImageEmbeddingModel()

    # Setup Image Procs
    procs = []
    toPipes = [mp.Queue() for i in range(numberOfThreads)]
    listenPipe = mp.Queue()
    wordsPerWorker = np.ceil(vocabSize / numberOfThreads)
    for i in range(numberOfThreads):
        procVocab = vocab[int(i * wordsPerWorker): int((i + 1) * wordsPerWorker)]
        procs.append(mp.Process(target=_imageWorker,
                                args=(i, procVocab, wordsPerBatch, imgSize, listenPipe, toPipes[i])))

    # Start all image procs
    for i, p in enumerate(procs):
        p.start()
        toPipes[i].put('Get Images')

    # Setup main pipeline
    filesCreated = 0
    totalEmbeddingsSaved = 0
    embeddingsInCurrentFile = 0
    currentFile = open(outFolder + "/Embeddings{}.txt".format(filesCreated), 'w')
    while (totalEmbeddingsSaved < vocabSize):
        id, words, imgs = listenPipe.get()

        try:
            for i, embedding in enumerate(EmbeddingModel.createAverageEmbeddingForImages(embeddingModel, imgs)):
                currentFile.write("{} {}\n".format(words[i], np.array2string(embedding)[1:-1]))

            embeddingsInCurrentFile += wordsPerBatch
            totalEmbeddingsSaved += wordsPerBatch
            print(totalEmbeddingsSaved, "/", vocabSize)

            if (embeddingsInCurrentFile >= wordsPerFile):
                currentFile.close()
                embeddingsInCurrentFile = 0
                filesCreated += 1
                currentFile = open(outFolder + "/Embeddings{}.txt".format(filesCreated), 'w')

        except Exception as e:
            print(e)

        toPipes[id].put("Get Images")

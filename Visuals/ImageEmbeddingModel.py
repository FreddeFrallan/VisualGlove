from Datasets import DatasetManager
import numpy as np


def createImageEmbeddingModel():
    import keras
    model = keras.applications.mobilenet_v2.MobileNetV2()
    outLayer = model.get_layer('global_average_pooling2d_1')
    return keras.Model(model.inputs, outLayer.output)


def createAverageEmbeddingForImages(model, wordImages):
    '''Images is a list containing lists of images per word'''
    numberOfImagesPerWord = [len(imgs) for imgs in wordImages]
    allImages = []
    for imgs in wordImages:
        allImages += imgs

    predictions = model.predict(np.array(allImages))
    vectorsPerWord = []
    counter = 0
    for n in numberOfImagesPerWord:
        vectorsPerWord.append(predictions[counter:counter + n])
        counter += n

    return [np.mean(np.array(wordVectors), axis=0) for wordVectors in vectorsPerWord]


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
    embeddingModel = createImageEmbeddingModel()

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
            for i, embedding in enumerate(createAverageEmbeddingForImages(embeddingModel, imgs)):
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


if __name__ == '__main__':
    from NormalGlove import Vocab
    path = DatasetManager.getVisualEmbeddingsFullSizeFolderPath()
    vocab = Vocab.readVocabFromCSVFile()
    createAverageEmbeddingsForVocab(vocab, 3, 10, [224, 224], path, 3)


    with open(path + "/Embeddings1", 'r') as f:
        for l in f.readlines():
            print(l[:40])

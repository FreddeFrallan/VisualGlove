import multiprocessing as mp
import numpy as np


############# Parsing


def _getUrlAndWordFromLine(line, seperatorChar='|'):
    splitPos = line.rfind(seperatorChar)
    if (splitPos < 0):
        return "", "", False
    return line[:splitPos].strip(), line[splitPos + 1:].strip(), True


def parseURLDatabase(urlFilePath):
    print("Loading URL's from:", urlFilePath)
    words = {}
    with open(urlFilePath, 'r', encoding='UTF-16') as file:
        readCounter = 0
        line = file.readline()
        while line:
            url, keyword, success = _getUrlAndWordFromLine(line)
            if (success):
                if (keyword not in words):
                    words[keyword] = []
                words[keyword].append(url)

            readCounter += 1
            line = file.readline()

    return words


#################################

def _downloadWorker(id, listenPipe, sendPipe, size, imgsPerWord):
    from io import BytesIO
    from PIL import Image
    import urllib3, logging

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.getLogger("urllib3").setLevel(logging.FATAL)
    client = urllib3.PoolManager()

    def downloadAndSizeImages(urls, size, timeout=2):
        images = []
        for url in urls:
            if (len(images) >= imgsPerWord):
                break
            try:
                response = client.request('GET', url, timeout=timeout)
                img = Image.open(BytesIO(response.data))
                img = img.convert('RGB')
                images.append(np.array(img.resize(size)))
            except:
                pass
        return images

    print("Starting download worker:", id)
    status, word, urls = listenPipe.get()
    while (status != 'quit'):
        imgs = downloadAndSizeImages(urls, size)
        sendPipe.put((id, word, len(imgs), imgs))
        status, word, urls = listenPipe.get()
    print("Quiting download worker:", id)

def _fileWorker(filePath, listenPipe):
    print("File Worker Started")
    with open(filePath, 'w', encoding='UTF-16') as outFile:
        status, word, data = listenPipe.get()
        while(status != "quit"):
            outFile.write("{} {}\n".format(word, " ".join(map(str, data))))
            status, word, data = listenPipe.get()


def embedImages():
    from Visuals import EmbeddingModel
    import time

    fileName = "VisualGlove-3.0 FullSize.txt"
    urlPath = "VisualGlove-2.0-Urls.txt"
    numberOfWorkers = 30
    imgsPerWord = 10
    size = (331, 331)
    #size = (224, 224)

    embeddingModel = EmbeddingModel.createImageEmbeddingModel()
    wordUrls = parseURLDatabase(urlPath)
    vocab = list(wordUrls.keys())
    vocabSize = len(vocab)
    print("Vocab Size:", vocabSize)

    filePipe = mp.Queue()
    fileProc = mp.Process(target=_fileWorker, args=(fileName, filePipe))
    fileProc.start()

    listenPipe = mp.Queue()
    procPipes = [mp.Queue() for i in range(numberOfWorkers)]
    procs = [mp.Process(target=_downloadWorker, args=(i, procPipes[i], listenPipe, size, imgsPerWord)) for i in
             range(numberOfWorkers)]

    for i, w in enumerate(vocab[:numberOfWorkers]):
        procs[i].start()
        procPipes[i].put(("", w, wordUrls[w]))

    sentWordsCounter = numberOfWorkers
    collectedWords = 0
    t1 = time.time()
    amounts = []
    #with open(fileName, 'w') as outFile:
    while (collectedWords < vocabSize):
        pID, word, amount, imgs = listenPipe.get()
        collectedWords += 1
        amounts.append(amount)

        if (amount > 0):
            x = embeddingModel.predict(np.array(imgs))
            filePipe.put(("", word, np.mean(x, axis=0)))

        if (sentWordsCounter < vocabSize):
            nextWord = vocab[sentWordsCounter]
            sentWordsCounter += 1
            procPipes[pID].put(("", nextWord, wordUrls[nextWord]))
        else:
            procPipes[pID].put(("quit", "", []))

        if (collectedWords % 10 == 0):
            avgSpeed = round((time.time() - t1) / collectedWords, 4)
            timeLeft = round(((vocabSize - collectedWords) * avgSpeed) / 3600, 4)
            print("{} / {}".format(collectedWords, vocabSize),
                  "  Speed(sec/w): {}   Time Left(H): {}".format(avgSpeed, timeLeft))

    '''

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
            # if (batchCounter % 100 == 0):
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
    '''


if (__name__ == '__main__'):
    embedImages()

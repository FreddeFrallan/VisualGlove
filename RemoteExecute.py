from Visuals import ImageEmbedder, Leftovers, VisualizeEmbeddings, PCAReduction
from NormalGlove import GloveFormatter, Model
from Datasets import DatasetManager
import os


def fixLeftovers():
    from NormalGlove import Vocab
    path = DatasetManager.getVisualEmbeddingsFullSizeFolderPath()
    vocab = Vocab.readVocabFromCSVFile()
    lostEmbeddings = Leftovers.findWordsThatLackEmbedings(vocab, path)
    print("Leftovers found", len(lostEmbeddings))
    ImageEmbedder.createAverageEmbeddingsForVocab(lostEmbeddings, 3, 10, [224, 224], path, 3)


def concatEmbeddingFiles():
    path = DatasetManager.getVisualEmbeddingsFullSizeFolderPath()
    GloveFormatter.concatenateEmbeddingsFiles(path, "VisualGlove-Full.txt")


def getTopSimilarityFromCombinedKeyedVectors(filePath, outFile, words, n=20):
    model = Model.loadGloveVectors(filePath)#Model.loadKeyedVectors(filePath)
    with open(outFile, 'w') as file:
        for w in words:
            file.write("**** {}\n".format(w))
            sim = model.most_similar(w, topn=n)
            for s in sim:
                file.write("{}\n".format(s[0]))
            for s in sim:
                file.write("{}\n".format(round(s[1], 3)))



if __name__ == '__main__':
    # createAverageEmbeddingsForVocab(vocab, 3, 10, [224, 224], path, 3)
    # if(sys.argv[0].lower() == "leftovers"):
    # fixLeftovers()
    # concatEmbeddingFiles()
    # VisualizeEmbeddings.main()
    # PCAReduction.main()

    sizes = [(300, 50), (300, 150), (300, 300), (300, 300), (100, 300), (50, 300)]
    for g, v in sizes:
        path = "/home/ubuntu/VisualGlove/Datasets/CombinedGlove/Glove{}-Visual{}/Keyed-Glove{}-Visual{}".format(g, v, g, v)
        m = Model.loadKeyedVectors(path)
        m.save_word2vec_format("Glove{}-Visual{}-W2V.txt".format(g, v))

    '''
    words = ["trump", "cat", 'yellow', 'banana', 'art', 'arm', 'blue', 'sky', 'dance', 'face', 'stake', 'forest',
             'drama', 'obama']
    pathBase = "/home/ubuntu/VisualGlove/Datasets/CombinedGlove/"
    sizes = [(300, 50), (300, 150), (300, 300), (300, 300), (100, 300), (50, 300)]
    #sizes = [(50, 150), (100, 200)]
    for gloveSize, visualGloveSize in sizes:
        keyFile = pathBase + "/Glove{}-Visual{}/Keyed-Glove{}-Visual{}".format(gloveSize, visualGloveSize, gloveSize, visualGloveSize)
    keyFile = "/home/ubuntu/VisualGlove/Datasets/StandardGlove/glove.6B.300d.txt"
    outFile = "Similarites-Glove{}-Visual{}.txt".format(300, 0)
    getTopSimilarityFromCombinedKeyedVectors(keyFile, outFile, words)

    keyFile = "/home/ubuntu/VisualGlove/Datasets/VisualEmbeddings/300/VisualGlove-300.txt"
    outFile = "Similarites-Glove{}-Visual{}.txt".format(0, 300)
    getTopSimilarityFromCombinedKeyedVectors(keyFile, outFile, words)
    '''

    '''
    file = "/home/ubuntu/VisualGlove/Datasets/VisualEmbeddings/300/" + "Keyed-VisualGlove-300.txt"
    print("Loading Word2Vec model from:", file)
    model = Model.loadKeyedVectors(file)
    print("Model Loaded")

    for w in ["yellow", "banana", "orange", "truck", "sky", "face"]:
        print(w)
        for d in model.most_similar(w):
            print(d)

    print(model.distance("yellow", "banana"))

    #sizes = [(50, 50), (50, 150)]
    #sizes = [(100, 100), (100, 200)]
    #sizes = [(300, 50), (300, 100), (300, 150), (300, 200), (300, 300)]
    sizes = [(50, 300), (100, 300)]
    for gloveSize, visualGloveSize in sizes:
        gloveFile = "/home/ubuntu/VisualGlove/Datasets/StandardGlove/glove.6B.{}d.txt".format(gloveSize)
        visualGloveFile = "/home/ubuntu/VisualGlove/Datasets/VisualEmbeddings/{}/VisualGlove-{}.txt".format(
            visualGloveSize, visualGloveSize)

        saveDir = "/home/ubuntu/VisualGlove/Datasets/CombinedGlove/Glove{}-Visual{}/".format(gloveSize, visualGloveSize)
        os.makedirs(saveDir)
        newFileName = saveDir + "CombinedGlove-{}-{}.txt".format(gloveSize, visualGloveSize)
        GloveFormatter.combineGloveFiles(visualGloveFile, gloveFile, newFileName)

        keyedFilename = "Keyed-Glove{}-Visual{}".format(gloveSize, visualGloveSize)
        GloveFormatter.createKeyedVectorsFromGloveFile(newFileName, keyedFilename)
    '''

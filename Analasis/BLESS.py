from Datasets import DatasetManager
from NormalGlove import Model
from Analasis import Utils


def generateWordCouples():
    couples = []
    with open(DatasetManager.getBLESSDataset(), 'r') as file:
        for l in file.readlines():
            word, _, type, word2 = l.lower().strip().split(' ')
            couples.append((word, word2, type))
    return couples


def evaluateEmbeddings(modelPaths, outputFilename, distance=True):
    couples = generateWordCouples()
    results = []
    for p in modelPaths:
        print(p)
        results.append(Utils.getTypeScoresForEmbeddings(couples, Model.loadKeyedVectors(p), distance=distance))
    Utils.storeResultsToDisk(results, modelPaths, outputFilename)


def evaluateEmbeddingPairs(modelPaths, outputFilename, distance=True):
    couples = generateWordCouples()
    results = []
    for p1, p2 in modelPaths:
        m1, m2 = Model.loadKeyedVectors(p1), Model.loadKeyedVectors(p2)
        print("Bless", m1, m2)
        results.append(Utils.getTypeScoresForEmbeddings(couples, m1, m2, distance=distance))
    Utils.storeResultsToDisk(results, ["{}--{}".format(p1, p2) for p1, p2 in modelPaths], outputFilename, pairs=True)


def main():
    pass
    '''
    evaluateEmbeddings(Utils.getAllModelPaths(), "BLESS-Similarity Results.txt", distance=False)

    baseStandard = "/home/ubuntu/VisualGlove/Datasets/StandardGlove/"
    baseVisual = "/home/ubuntu/VisualGlove/Datasets/VisualEmbeddings/"
    paths = []
    for d in [50, 100, 200, 300]:
        visual = baseVisual + "{}/Keyed-VisualGlove-{}".format(d, d)
        paths.append((visual, baseStandard + "Keyed-Glove{}-Visual0".format(d)))

    evaluateEmbeddingPairs(paths, "BLESS-Similarity Pair Results.txt", distance=False)
    '''


if (__name__ == '__main__'):
    main()

# 300, 50
# {'attri': 354.57167130748513, 'coord': 1519.4650538323256, 'event': 531.2666583687629, 'hyper
# ': 284.66584216609823, 'mero': 543.3861471953717, 'random': 578.9511052384229}

# 300, 300
# {'attri': 327.77185295893577, 'coord': 1425.2868632502395, 'event': 487.8749753979832, 'hyper
# ': 266.2803596403974, 'mero': 504.25721478718054, 'random': 519.6760132782766}

# 50, 300
# {'attri': 434.74813186131684, 'coord': 1609.6085274307713, 'event': 598.2692101499282, 'hyper
# ': 313.5659697507555, 'mero': 600.0812845774522, 'random': 734.6452639473853}

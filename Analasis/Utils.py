from Datasets import DatasetManager
import numpy as np


def storeResultsToDisk(results, modelNames, fileName, pairs=False):
    with open(fileName, 'w') as file:
        for i, modelName in enumerate(modelNames):
            file.write("**************\n")
            file.write("{}\n".format(modelName))
            file.write("**************\n")

            result = results[i]
            for k in result.keys():
                file.write("{}:\n".format(k))
            for k in result.keys():
                file.write("{}\n".format(round(result[k][0] / result[k][1], 3)))

            if (pairs):
                file.write("**** Ratio Model-1\n")
                for k in result.keys():
                    score, amount, m1, m2 = result[k]
                    file.write("{}\n".format(round(m1 / amount, 3)))


def getTopScore(model1, model2, word1, word2, distance=True):
    if (distance):
        results = [model1.distance(word1, word2), model2.distance(word1, word2)]
        return np.min(results), np.argmin(results)
    else:
        results = [model1.similarity(word1, word2), model2.similarity(word1, word2)]
        return np.max(results), np.argmax(results)


def getTypeScoresForEmbeddings(wordCouples, model, model2=None, distance=True):
    typeScores = {}
    errorWords = []
    for w1, w2, type in wordCouples:
        try:
            if (type not in typeScores):
                typeScores[type] = [0, 0] if model2 == None else [0, 0, 0, 0]

            if (model2 != None):
                score, usedModel = getTopScore(model, model2, w1, w2, distance=distance)
                typeScores[type][0] += score
                typeScores[type][2 + usedModel] += 1
            elif (distance):
                typeScores[type][0] += model.distance(w1, w2)
            else:
                typeScores[type][0] += model.similarity(w1, w2)

            typeScores[type][1] += 1

        except Exception as e:
            errorWords.append((w1, w2))

    print("Words not counted:", len(errorWords))
    return typeScores


def getNormalGlovePaths():
    return [DatasetManager.getNormalGloveFolderPath() + "/Keyed-Glove{}-Visual0".format(i) for i in [50, 100, 200, 300]]


######################### Top 100K
def getCombinedTop400KPaths():
    sizes = [(50, 50), (50, 150), (100, 100), (200, 100), (100, 200)]
    folderPath = DatasetManager._getDatasetsFolderPath() + "/CombinedGlove/"
    return [folderPath + "/Glove{}-Visual{}/Keyed-Glove{}-Visual{}".format(g, v, g, v) for g, v in sizes]


def getTop400KVisualOnly():
    basePath = DatasetManager.getVisualEmbeddingsFolderPath()
    return [basePath + "{}/Keyed-VisualGlove-{}".format(i, i) for i in [50, 100, 200, 300]]


def getAllTop400KAndNormal():
    return getNormalGlovePaths() + getTop400KVisualOnly() + getCombinedTop400KPaths()


######################### Top 100K

def getTop100KVisualOnly():
    basePath = DatasetManager.getVisualEmbeddingsFolderPath()
    return [basePath + "Top-100K-{}/Keyed-VisualGlove-{}".format(i, i) for i in [50, 100, 200, 300]]


def getTop100KCombinedPaths():
    sizes = [(50, 50), (50, 150), (100, 100), (200, 100), (100, 200)]
    folderPath = DatasetManager._getDatasetsFolderPath() + "/CombinedGlove-Top100K/"
    return [folderPath + "/Glove{}-Visual{}/Keyed-Glove{}-Visual{}".format(g, v, g, v) for g, v in sizes]


def getAllTop100KAndNormal():
    return getNormalGlovePaths() + getTop100KVisualOnly() + getTop100KCombinedPaths()


######################### Top & Skip

def getTopAndSkipKVisualOnly(top, skip, version=1):
    basePath = DatasetManager.getVisualEmbeddingsFolderPath(version)
    return [basePath + "Top{}K-Skip{}-{}/Keyed-VisualGlove-{}".format(top, skip, i, i) for i in [50, 100, 200, 300]]


def getTopAndSkipCombined(top, skip, version=1):
    sizes = [(50, 50), (50, 150), (100, 100), (200, 100), (100, 200), (50, 300), (100, 300), (300, 50), (300, 150),
             (300, 300)]
    if (version == 1):
        basePath = DatasetManager._getDatasetsFolderPath() + "/CombinedGlove-Top{}K-Skip{}/".format(top, skip)
    else:
        basePath = DatasetManager._getDatasetsFolderPath() + "/V{}/CombinedGlove-Top{}K-Skip{}/".format(version, top,
                                                                                                        skip)
    return [basePath + "Glove{}-Visual{}/Keyed-Glove{}-Visual{}".format(g, v, g, v) for g, v in sizes]


def getAllTopAndSkip(top, skip, version=1):
    return getNormalGlovePaths() + getTopAndSkipKVisualOnly(top, skip, version) + getTopAndSkipCombined(top, skip,
                                                                                                        version)

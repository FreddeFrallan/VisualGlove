from Datasets import DatasetManager
from Analasis import Utils
from NormalGlove import Model


def _getDataset(filepath=None):
    if (filepath == None):
        filepath = DatasetManager.getSynonymsDataset()

    with open(filepath, 'r') as file:
        sections = file.read().split('=')

    parsedSections = []
    for s in sections:
        subSections = s.strip().replace('\n', ' ').replace('.', ' ').split(':'), "\n"

        sortedSection = {}
        currentSection = ""
        for sub in subSections:
            for w in ' '.join(sub).split(' '):
                w = w.strip()
                if (w == "KEY"):
                    currentSection = "KEY"
                    sortedSection[w] = []
                elif (w == "SYN"):
                    currentSection = "SYN"
                    sortedSection[w] = []
                elif (w == "ANT"):
                    currentSection = "ANT"
                    sortedSection[w] = []

                elif (currentSection != "" and len(w) > 0 and str.isspace(w) == False):
                    sortedSection[currentSection].append(w.lower())

        if ('KEY' not in sortedSection or len(sortedSection['KEY']) == 0):
            continue
        if (('SYN' in sortedSection and len(sortedSection['SYN']) > 0) or
                ('ANT' in sortedSection and len(sortedSection['ANT']) > 0)):
            parsedSections.append(sortedSection)

    return parsedSections


def generateWordCouples(datasetPath=None):
    couples = []
    for d in _getDataset(filepath=datasetPath):
        w = d['KEY'][0]
        if ('SYN' in d):
            for w2 in d['SYN']:
                couples.append((w, w2, 'SYN'))
        if ('ANT' in d):
            for w2 in d['ANT']:
                couples.append((w, w2, 'ANT'))
    return couples


def evaluateModels(modelPaths, outputFilename, datasetPath=None, distance=False):
    couples = generateWordCouples(datasetPath)
    results = []
    for mPath in modelPaths:
        print("***Synonyms***\n", mPath)
        model = Model.loadKeyedVectors(mPath)
        results.append(Utils.getTypeScoresForEmbeddings(couples, model, distance=distance))

    Utils.storeResultsToDisk(results, modelPaths, outputFilename)


def main():
    pass
    # evaluateAllModels()


if (__name__ == '__main__'):
    main()

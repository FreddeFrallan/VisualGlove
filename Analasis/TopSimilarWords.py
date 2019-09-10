from NormalGlove import Model

STANDARD_WORDS = ["trump", "cat", 'yellow', 'banana', 'art', 'arm', 'blue', 'sky', 'dance', 'face', 'stake', 'forest',
                  'drama', 'obama']


def getTopSimilarityFromCombinedKeyedVectors(paths, outFile, words=None, n=20):
    if (words == None):
        words = STANDARD_WORDS
    with open(outFile, 'w', encoding='utf-8') as file:
        for modelP in paths:
            model = Model.loadKeyedVectors(modelP)
            file.write("**** ||||| {}\n".format(modelP))
            for w in words:
                file.write("**** {}\n".format(w))
                sim = model.most_similar(w, topn=n)
                for s in sim:
                    file.write("{}\n".format(s[0]))
                for s in sim:
                    file.write("{}\n".format(round(s[1], 3)))

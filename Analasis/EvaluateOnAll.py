from Analasis import Utils, BLESS, Synonyms, TopSimilarWords


def main():
    version = 2
    top = 400
    skip = 0
    name = "Top{}-Skip{}-V{}".format(top, skip, version)
    paths = Utils.getAllTopAndSkip(top, skip, version)
    paths.append("/home/ubuntu/VisualGlove/Datasets/V2/VisualEmbeddings/FullSize/Keyed-VisualGlove-Full")

    TopSimilarWords.getTopSimilarityFromCombinedKeyedVectors(paths, "{}-TopSimilair Results.txt".format(name))
    for distance in [True, False]:
        distText = "Distance" if distance else "Similarity"
        BLESS.evaluateEmbeddings(paths, "{}-BLESS-{} Results.txt".format(name, distText), distance=distance)
        Synonyms.evaluateModels(paths, "{}-Synonym-{} Results.txt".format(name, distText), distance=distance)


    print("Evaluation is finished\n")

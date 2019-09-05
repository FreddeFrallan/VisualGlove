from Datasets import DatasetManager
import csv


def createVocabCSVFile(fileName="VisualGloveVocab.csv"):
    from NormalGlove import Model
    model = Model.loadGlove50()
    vocab = list(model.wv.vocab)
    with open(fileName, 'w') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(vocab)


def readVocabFromCSVFile():
    with open(DatasetManager.getGloveVocabCSVPath(), 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return list(reader)[0]


def createVocabScrapeFile(fileName):
    vocab = readVocabFromCSVFile()
    with open(fileName, 'w') as file:
        for w in vocab:
            file.write(w)
            file.write("\n")


if (__name__ == '__main__'):
    createVocabScrapeFile("GloveVocabScrapeFile.txt")

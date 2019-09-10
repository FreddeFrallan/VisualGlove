from Datasets import DatasetManager
import csv


def _loadVocab():
    with open(DatasetManager.getGloveVocabCSVPath(), 'r', encoding='utf-8') as f:
        vocab = list(csv.reader(f))[0]
    return vocab


def _filterWord(word):
    if (str.isalpha(word[0]) == False):  # Starts on non letter
        return False

    for c in word:  # Contains number
        if (str.isnumeric(c)):
            return False

    return True


def filterNonAlphaNumerical(outFilePath):
    vocab = _loadVocab()
    filteredVocab = [w for w in vocab if _filterWord(w)]
    with open(outFilePath, 'w', encoding='utf-8') as file:
        for w in filteredVocab:
            file.write("{}\n".format(w.strip()))

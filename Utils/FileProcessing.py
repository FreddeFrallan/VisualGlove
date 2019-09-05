import pickle


def saveToFile(data, filename="Temp.data"):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def loadFromFile(fileName="Temp.data"):
    with open(fileName, 'rb') as file:
        return pickle.load(file)

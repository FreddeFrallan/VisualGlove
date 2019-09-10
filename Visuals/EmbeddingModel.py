import numpy as np


def createImageEmbeddingModel():
    import keras, kito
    model = keras.applications.MobileNetV2()
    model.summary()
    outLayer = model.get_layer('global_average_pooling2d_1')
    #return keras.Model(model.inputs, outLayer.output)
    return kito.reduce_keras_model(keras.Model(model.inputs, outLayer.output))


def createAverageEmbeddingForImages(model, wordImages):
    '''Images is a list containing lists of images per word'''
    numberOfImagesPerWord = [len(imgs) for imgs in wordImages]
    allImages = []
    for imgs in wordImages:
        allImages += imgs

    predictions = model.predict(np.array(allImages))
    vectorsPerWord = []
    counter = 0
    for n in numberOfImagesPerWord:
        vectorsPerWord.append(predictions[counter:counter + n])
        counter += n

    return [np.mean(np.array(wordVectors), axis=0) for wordVectors in vectorsPerWord]

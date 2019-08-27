
def _createModelFromSettings():
    pass

def createModel(wordEmbeddingSize, vocabSize, vocabEmbeddings=None, maxSentenceLen=50):
    from keras import Model, layers
    import keras

    encoderIn = layers.Input(shape=(maxSentenceLen, ))
    encoderEmbedding = layers.Embedding(vocabSize, wordEmbeddingSize, name="EncodeEmbedding")(encoderIn)
    encoder = layers.LSTM(wordEmbeddingSize, return_state=True, return_sequences=False)
    encoderOut, stateH, stateC = encoder(encoderEmbedding)
    encoderStates = [stateH, stateC]

    decoderIn = layers.Input(shape=(maxSentenceLen, ))
    decoderEmbedding = keras.layers.Embedding(vocabSize, wordEmbeddingSize, name="DecodeEmbedding")(decoderIn)
    decoder = layers.LSTM(wordEmbeddingSize, return_sequences=True, return_state=True)
    decoderOut, _, _ = decoder(decoderEmbedding, initial_state= encoderStates)
    decodeDense = layers.TimeDistributed(layers.Dense(vocabSize, activation='softmax'))
    decoderOut = decodeDense(decoderOut)

    return Model([encoderIn, decoderIn], decoderOut)

model = createModel(300, 4000)
model.summary()
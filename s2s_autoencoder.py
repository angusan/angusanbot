from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, RepeatVector
from keras import backend as K
from keras import regularizers
from keras.regularizers import L1L2, l2, l1
import numpy as np

num_samples = 10000  # Number of samples to train on.
def parseTextCorpus(data_path):
    input_texts = []
    target_texts = []
    read_characters = set()
    intents = set()
    # for i in range(int('0x4E00', 16), int('0x9FA5', 16) + 1):
    #    read_characters.add(chr(i))
    #    read_characters.add(chr(i))
    lines = open(data_path, encoding='utf-8').read().split('\n')
    texts_intent = {}
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text, intent = line.split(',')
        input_texts.append(input_text)
        target_texts.append(target_text)
        texts_intent[input_text] = intent
        for char in input_text:
            if char not in read_characters:
                read_characters.add(char)
        for char in target_text:
            if char not in read_characters:
                read_characters.add(char)
        if intent not in intents:
            intents.add(intent)

    corpus_meta = {}
    read_characters = sorted(list(read_characters))
    intents = sorted(list(intents))
    corpus_meta["input_texts"] = input_texts
    corpus_meta["target_texts"] = target_texts
    corpus_meta["characters_dict"] = dict([(char, i) for i, char in enumerate(read_characters)])
    corpus_meta["intent_dict"] = dict([(_intent, i) for i, _intent in enumerate(intents)])
    corpus_meta["texts_intent_dict"] = texts_intent

    return corpus_meta

corpus_meta = parseTextCorpus('intent.txt')
num_tokens = len(corpus_meta.get("characters_dict"))
max_sentence_length = max([len(txt) for txt in corpus_meta.get("input_texts")])
num_intents = len(corpus_meta.get("intent_dict"))
sample_size = len(corpus_meta.get("input_texts"))

print('Number of samples:', len(corpus_meta.get("input_texts")))
print('Number of unique output tokens:', num_tokens)
print('Max sequence length for inputs:', max_sentence_length)
print('Number of intents:', num_intents)

encoder_input_data = np.zeros((sample_size, max_sentence_length, num_tokens), dtype='float32')
decoder_target_data = np.zeros((sample_size, max_sentence_length, num_tokens), dtype='float32')
target_intent_data = np.zeros((sample_size, num_intents))

for i, (input_text, target_text) in enumerate(zip(corpus_meta.get("input_texts"), corpus_meta.get("target_texts"))):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, corpus_meta.get("characters_dict").get(char)] = 1.
    for t, char in enumerate(target_text):
        decoder_target_data[i, t, corpus_meta.get("characters_dict").get(char)] = 1.
    setence_intent = corpus_meta.get("texts_intent_dict").get(input_text);
    intent_index = corpus_meta.get(setence_intent)
    target_intent_data[i, intent_index] = 1.


latent_dim = 150  # Latent dimensionality of the encoding space.
K.set_learning_phase(True)
inputs = Input(shape=(max_sentence_length, num_tokens))
encoded = Bidirectional(LSTM(latent_dim))(inputs)
encoded = Dense(latent_dim, activation='relu')(encoded)
# repeat for output
decoded = RepeatVector(max_sentence_length)(encoded)
decoded = LSTM(num_tokens, return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(num_tokens, activation='softmax'))(decoded)
autoencoder = Model(inputs, outputs)

early_stopping = EarlyStopping(monitor='val_acc', patience=50)
model_check_point = ModelCheckpoint('s2s_best_autoencoder.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)
autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
autoencoder.summary()

batch_size = 1  # Batch size for training.
epochs = 5000  # Number of epochs to train for.
autoencoder.fit(encoder_input_data, decoder_target_data,
                batch_size=batch_size, epochs=epochs, validation_split=0.2,
                callbacks=[early_stopping, model_check_point])

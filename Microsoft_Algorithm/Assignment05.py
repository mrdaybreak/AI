# What is autoencoder?
print('自编码器autoencoder是一种无监督的学习算法，主要用于数据的降维或者特征的抽取')

# What are the differences between greedy search and beam search?
print('greedy search贪心搜索选择每个输出的最大概率，直到出现终结符或者最大句子长度；beam search集束搜索重复使用概率最大beam size在生成词时获取序列，直到出现终结符')

# What is the intuition of attention mechanism?
print('一个单词聚焦所有语句的情况，一个加权求和')

# What is the disadvantage of word embeding introduced in previous lectures?
print('每个词都是孤立的，使得算法对相关词对泛化性不强')

# Briefly describe what is self-attention and what is multi-head attention?
print('self-attention,Query、Key和Value的向量表示均来自于同一输入文本; multi-head attention利用不同的Self-Attention模块获得文本中每个字在不同语义空间下的增强语义向量，并将每个字的多个增强语义向量进行线性组合，从而获得一个最终的与原始字向量长度相同的增强语义向量')

from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from keras.utils import plot_model
import pandas as pd
import numpy as np


def create_model(input, output, units):
    encoder_input = Input(shape=(None, input))
    encoder = LSTM(units, return_state=True)
    _, encoder_h, encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h, encoder_c]

    decoder_input = Input(shape=(None, output))
    decoder = LSTM(units, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)
    decoder_dense = Dense(output, activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)

    encoder_infer = Model(encoder_input, encoder_state)

    decoder_state_input_h = Input(shape=(units,))
    decoder_state_input_c = Input(shape=(units,))

    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]

    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input, initial_state=decoder_state_input)

    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]
    decoder_infer_output = decoder_dense(decoder_infer_output)

    decoder_infer = Model([decoder_input] + decoder_state_input, [decoder_infer_output] + decoder_infer_state)

    return model, encoder_infer, decoder_infer

N_UNITS = 256
BATCH_SIZE = 64
EPOCH = 50
NUM_SAMPLES = 10000

data_path = 'cmn-eng/cmn.txt'
df = pd.read_table(data_path, header=None).iloc[:NUM_SAMPLES, :,]
df.columns = ['inputs', 'targets', 'others']

df['targets'] = df['targets'].apply(lambda x: '\t'+x+'\n')

input_texts = df.inputs.values.tolist()
target_texts = df.targets.values.tolist()

input_characters = sorted(list(set(df.inputs.unique().sum())))
target_characters = sorted(list(set(df.targets.unique().sum())))

INPUT_LENGTH = max([len(i) for i in input_texts])
OUTPUT_LENGTH = max([len(i) for i in target_texts])
INPUT_FEATURE_LENGTH = len(input_characters)
OUTPUT_FEATURE_LENGTH = len(target_characters)

encoder_input = np.zeros((NUM_SAMPLES, INPUT_LENGTH, INPUT_FEATURE_LENGTH))
decoder_input = np.zeros((NUM_SAMPLES, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH))
decoder_output = np.zeros((NUM_SAMPLES, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH))

input_dict = {char:index for index, char in enumerate(input_characters)}
input_dict_reverse = {index:char for index, char in enumerate(input_characters)}
target_dict = {char:index for index, char in enumerate(target_characters)}
target_dict_reverse = {index:char for index, char in enumerate(target_characters)}

for seq_index, seq in enumerate(input_texts):
    for char_index, char in enumerate(seq):
        encoder_input[seq_index, char_index, input_dict[char]] = 1

for seq_index,seq in enumerate(target_texts):
    for char_index,char in enumerate(seq):
        decoder_input[seq_index,char_index, target_dict[char]] = 1.0
        if char_index > 0:
            decoder_output[seq_index,char_index-1, target_dict[char]] = 1.0


model_train, encoder_infer, decoder_infer = create_model(INPUT_FEATURE_LENGTH, OUTPUT_FEATURE_LENGTH, N_UNITS)
model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model_train.summary()

validation_split = 0.2
model_train.fit([encoder_input, decoder_input], decoder_output, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=validation_split)
model_train.save('transform.h5')

def predict_chinese(source, encoder_inference, decoder_inference, n_steps, features):
    state = encoder_inference.predict(source)
    predict_seq = np.zeros((1, 1, features))
    predict_seq[0, 0, target_dict['\t']] = 1

    output = ''

    for i in range(n_steps):
        yhat, h, c = decoder_inference.predict([predict_seq]+state)
        char_index = np.argmax(yhat[0, -1, :])
        char = target_dict_reverse[char_index]
        output += char
        state = [h, c]
        predict_seq = np.zeros((1, 1, features))
        predict_seq[0, 0, char_index] = 1
        if char == '\n':
            break
    return output

for i in range(1000,1100):
    test = encoder_input[i:i+1,:,:]
    out = predict_chinese(test,encoder_infer,decoder_infer,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH)
    print(input_texts[i])
    print(out)










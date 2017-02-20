from __future__ import print_function
from keras.callbacks import History 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os


sub_dir = 'Obas Machadianas' #Diretório no qual estão os aquivos de texto
file_list= os.listdir(sub_dir) #Lista os aquivos nesse diretório
text = '' # Variável a qual conterá uma síntese de todos os textos
for file in file_list:
    if(file[len(file)-4:len(file)] == '.txt'): # Abre apenas arquivos '.txt'
        text = text + open(os.path.join(sub_dir, file), 'r', encoding='utf-8-sig').read() #Concatena tais arquivos
    
chars = sorted(list(set(text))) # Cria uma lista ordenada com os caracteres presentes nos textos
print('total chars:', len(chars)) # Diz o número total de caracteres diferentes nestes - 155 nas obras machadianas
char_indices = dict((c, i) for i, c in enumerate(chars)) # Cria uma função que diz qual o número correspondente a um caracter na lista criada
indices_char = dict((i, c) for i, c in enumerate(chars)) # Cria outra função que diz qual o numero de um caracter na lista criada

# Seleciona maxlen caracteres no texto, salva essa sentença no array sentenças e coloca o próximo caracter em outro array
# Este caracter serve como a saida esperada quando a entrada for os 40 caracteres, ou seja, a previsão correta
# Após isso converte estes caracteres em uma matriz de tal forma que cada caracter tem um array cujos valores são 0 exceto pelo espaço
#correspondente ao em questão char na lista de chars, espaço o qual será 1, criando o array X de dados de entrada e o array Y de dados
#esperados na saída.
maxlen = 60
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step): # Corta o texto em sequencias de 40 caracter as quais se diferenciam das próximas por 1 caracter
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...') # Cria o vetor equivalente às sequencias X e Y equivalente caracteres esperados, no formato '0 e 1'

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    


print('Build model...') # Caso não queira carregar nenhum arquivo, constrói a rede
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

#Configurações para o Treinamento
optimizer = 'adam' #RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Treina o modelo, o salvando e mostrando um exemplo a cada iteração
print()

# define the checkpoint
filepath="weights-improvement2-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(X, y, batch_size=128, nb_epoch=60, validation_split=0.2 ) # Treina o modelo de acorodo com os argumentos em 'model.compile', usando os dados de entrada X e
                                                #y esperados como saída. Treina batch_size vezes antes de atualizar o pesos e o faz para X e Y
                                                #nb_epoch vezes.
file_name = 'Machado_training2.h5' # Escreve o nome do arquivo a ser salvo
print('Saving current state as: ' + file_name);
model.save('saved\\' + file_name); # Salva o arquivo


print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
from __future__ import print_function
from keras.callbacks import History 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os


train_name = 'Machado_Train7'
last_epoch = 18


sub_dir = 'Obas Machadianas' #Diretório no qual estão os aquivos de texto
file_list= os.listdir(sub_dir) #Lista os aquivos nesse diretório
text = '' # Variável a qual conterá uma síntese de todos os textos
for file in file_list:
    if(file[len(file)-4:len(file)] == '.txt'): # Abre apenas arquivos '.txt'
        text = text + open(os.path.join(sub_dir, file), 'r', encoding='utf-8-sig').read() #Concatena tais arquivos
    
chars = sorted(list(set(text))) # Cria uma lista ordenada com os caracteres presentes nos textos
print('total chars:', len(chars)) # Diz o número total de caracteres diferentes nestes - 120 com o texto normal e 82 com todas as letras maiúsculas
char_indices = dict((c, i) for i, c in enumerate(chars)) # Cria uma função que diz qual o número correspondente a um caracter na lista criada
indices_char = dict((i, c) for i, c in enumerate(chars)) # Cria outra função que diz qual o numero de um caracter na lista criada

sub_dir = 'Obas Machadianas\\Part2'
file_list= os.listdir(sub_dir)
text = ''
for file in file_list:
    if(file[len(file)-4:len(file)] == '.txt'): # Abre apenas arquivos '.txt'
        text = text + open(os.path.join(sub_dir, file), 'r', encoding='utf-8-sig').read() #Concatena tais arquivos

# Seleciona maxlen caracteres no texto, salva essa sentença no array sentenças e coloca o próximo caracter em outro array
# Este caracter serve como a saida esperada quando a entrada for os 40 caracteres, ou seja, a previsão correta
# Após isso converte estes caracteres em uma matriz de tal forma que cada caracter tem um array cujos valores são 0 exceto pelo espaço
#correspondente ao em questão char na lista de chars, espaço o qual será 1, criando o array X de dados de entrada e o array Y de dados
#esperados na saída.
maxlen = 100
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
    
loadname = train_name + '_Epoch-'+ str(last_epoch) + '.h5'
model = load_model('saved\\' + train_name + '\\' + loadname) # Carrega o modelo salvo, tanto os pesos quanto a própria estrutura da rede

#Configurações para o Treinamento
optimizer = 'adam'
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print()

plot_loss = []
plot_val_loss = []

f = open('saved\\' + train_name + '\\'  + train_name + '.txt', 'r')
data = f.read().split('\n')

data = iter(data)
next(data)
for datum in data:
    if(len(datum) > 4):
        num = datum.split('\t')
        if(int(num[0]) > last_epoch):
            break
        plot_loss.append(float(num[1]))
        plot_val_loss.append(float(num[2]))
    
f.close()

total_epochs = 40

for epoch in range(last_epoch + 1, total_epochs + last_epoch):
    history = model.fit(X, y, batch_size=128, nb_epoch=1, validation_split=0.2)
    plot_loss.append(history.history['loss'][0])
    plot_val_loss.append(history.history['val_loss'][0])
    file_name = train_name + '_Epoch-' + str(epoch) # Escreve o nome do arquivo a ser salvo
    print('Saving current state as: ' + file_name)
    model.save('saved\\' + train_name + '\\'  + file_name + 'PART2.h5'); # Salva o arquivo
    f = open('saved\\' + train_name + '\\' + train_name + 'PART2.txt', 'a')
    f.write(str(epoch) + '\t' + str(history.history['loss'][0]) + '\t' + str(history.history['val_loss'][0]) + '\n')
    f.close()

# summarize history for loss
plt.plot(plot_loss)
plt.plot(plot_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('saved\\' + train_name + '\\'  + train_name + 'PART2.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
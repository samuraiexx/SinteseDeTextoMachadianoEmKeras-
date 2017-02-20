from __future__ import print_function
from keras.callbacks import History 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys
import os

maxlen = 100
folder_name = 'Machado_Train1'
loadname = 'Machado_Train1_Epoch-13.h5'

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

# Esta função insere a influência da temperatura de modo que a partir de um vetor, o qual 
#contém a propabilidade de cada char poder ser o próximo da lista em cada uma das posições,
#eleva cada probabilidade à 1 sobre a temperatura e normaliza as probabilidades novamente.
# Após feito isso 'sorteia' o próximo caracter levando em consideração as probabilidades
#-como se jogasse um dado.
# Retorna a posição do caracter de saída
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64') # Transforma preds de numpy array em array
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) # Normaliza as probabilidades
    probas = np.random.multinomial(1, preds, 1) #'Joga o dado'
    return np.argmax(probas)


model = load_model('saved\\' + folder_name + '\\' + loadname) # Carrega o modelo salvo, tanto os pesos quanto a própria estrutura da rede

for diversity in [0.2, 0.5, 1.0, 1.2]: # Diversidade ou temeperatura, quanto maior, maior a aleatoriede dos caracteres
    print()
    print('----- diversity:', diversity)

    generated = '' # Cria a variável a qual armazenará o texto gerado
    sentence = 'Camargo era pouco simpático à primeira vista. Tinha as feições duras e frias, os olhos perscrutadores e sagazes, de uma sagacidade incômoda para quem encarava com eles, o que o não fazia atraente.' # A rede escreve o texto partindo desta frase seed
    generated += sentence
    sys.stdout.write(generated) # Escreve o inicio da sentença, só a seed, sem pular linha - por isso nao usa a funcao print()
    
    sentence = sentence[len(sentence) - maxlen:len(sentence)]
    
    for i in range(400): # Escreve 400 caracteres
        x = np.zeros((1, maxlen, len(chars))) # Cria a matriz de zeros e uns, tal qual foi feito para o texto
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1

        preds = model.predict(x, verbose=0)[0]  # Retorna o np_array com as probabilidades de cada char
        next_index = sample(preds, diversity) # Escolhe o próximo char usando a função sample, explicada a cima
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
from __future__ import print_function
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
    

# Constrói ou carrega o modelo
it_0 = 0 # Numero de iterações inicial, 0 caso não carregue nenhum arquivo
print(' Do you want to load a model? (y\\n)')
if(input() == 'y'):
    it_num_str = '';
    print(' Insert the file name')
    file_name = input()
    if(file_name.find('Machado_it-') and file_name.find('_slen-')): # Confere se o nome do arquivo confere com os padrões esprados
        print('O Nome do arquivo inserido não está de acordo com o padão criado')
        sys.exit(0)
    if(file_name[len(file_name)-3:len(file_name)] != '.h5'): # Adiciona a extenção, caso não o tenha sido feito
        file_name.join('.h5')
    try:
        model = load_model('saved\\' + file_name) # Carrega o modelo salvo, tanto os pesos quanto a própria estrutura da rede
    except Exception as e:
        print("Something went terribly wrong...I coudnt open the file..")
        sys.exit(0)
    for i, c in enumerate(file_name): # Salva o número da ulltima iteração do arquivo salvo, a partir do nome deste
        if(c.isdigit()):
            it_num_str = it_num_str + c
        if(c == '_' and file_name[i:i+5] == '_slen'):
            break
    it_0 = int(it_num_str)
else:
    print('Build model...') # Caso não queira carregar nenhum arquivo, constrói a rede
    model = Sequential()
    model.add(LSTM(512, input_shape=(maxlen, len(chars), return_sequences=True)))
    model.add(LSTM(512, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

#Configurações para o Treinamento
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


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

# Treina o modelo, o salvando e mostrando um exemplo a cada iteração
for iteration in range(it_0 + 1, 1000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1) # Treina o modelo de acorodo com os argumentos em 'model.compile', usando os dados de entrada X e
                                                #y esperados como saída. Treina batch_size vezes antes de atualizar o pesos e o faz para X e Y
                                                #nb_epoch vezes.
    
    file_name = 'Machado_it-' + str(iteration) + '_slen-' + str(maxlen) + '.h5' # Escreve o nome do arquivo a ser salvo
    print('Saving current state as: ' + file_name);
    model.save('saved\\' + file_name); # Salva o arquivo
    
    #start_index = random.randint(0, len(text) - maxlen - 1) #Função útil caso se queira usar partes das próprias obras como semente    

    for diversity in [0.2, 0.5, 1.0, 1.2]: # Diversidade ou temeperatura, quanto maior, maior a aleatoriede dos caracteres
        print()
        print('----- diversity:', diversity)

        generated = '' # Cria a variável a qual armazenará o texto gerado
        sentence = '— Morre-se. Quem não padece estas dores não as pode avaliar. O golpe foi profundo, e o meu coração é pusilânime; por mai' # A rede escreve o texto partindo desta frase seed
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated) # Escreve o inicio da sentença, só a seed, sem pular linha - por isso nao usa a funcao print()

        for i in range(400): # Escreve 400 caracteres.
            x = np.zeros((1, maxlen, len(chars))) # Cria a matriz de zeros e uns, tal qual foi feito para o texto
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]  # Retorna o np_array com as probabilidades de cada char
            next_index = sample(preds, diversity) # Escolhe o próximo char usando a função sample, explicada a cima
            next_char = indices_char[next_index]
            
            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
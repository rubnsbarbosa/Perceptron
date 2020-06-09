## Perceptron Simples

Em 1958, surgiu o primeiro algoritmo de RNAs proposto por Frank Rosenblatt, chamado Perceptron Simples. Esse algoritmo é um classificador linear binário, ou seja, classifica dados linearmente separáveis em apenas duas classes. 

De uma forma geral, o Perceptron Simples consiste do neurônio de McCulloch & Pitts (1943) combinado com uma regra de aprendizagem. A inteligência surge justamente da capacidade de aprender adicionada através desta regra.

## Algoritmo do Perceptron Simples

* Início (i = 0)
    * Definir o valor de *eta* entre 0 e 1;
    * Iniciar o peso com valores nulos ou randômicos;
* Funcionamento
    * Pegar os valores de entrada **X**(*i*);
    * Calcular a função de ativação *u*(*i*) [geralmente usamos a função degrau];
    * Calcular a saída *y*(*i*);
* Treinamento
    * Calcular o erro: **e**(*i*) = *y*(*i*) - *y_hat*(*i*);
    * Ajustar os pesos e bias através da regra de aprendizagem

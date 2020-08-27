# Keras_test_images
Testing Keras with images

O programa Keras_t.py para a base de Train, Test e Valid:
1 - carrega imagens 200 x 200 da base simpsons 
2 - utiliza Keras VGG16 model para extrair características
3 - Grava arquivo csv com classe, caracteristica1, caracteristica2, ...., caracteristicaN
4 - converte o arquivo csv para o formado libsvm atraves do programa csv2libsvm.py
5 - carrega os dados no fromato libsvm e separa X de y
6 - seleciona as melhores características usando RandomForestClassifier e SelectFromModel
7 - Chama os classificadores NB, DT e Knn.

Problema: o numero de características resultante é diferente nas bases Train, Test e Valid. 
Erro: ValueError: Expected input with 414 features, got 251 instead

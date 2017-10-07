from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_svmlight_file
import os
import csv
import glob
from skimage import data
import re
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


model = VGG16(weights='imagenet', include_top=False)  # prepara o modelo

img_path = '/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/Test_r/bart116_resized.jpg'

dir_base = '/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/'

dir_base_train = '/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/Train/'
dir_base_test = '/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/Test/'

dir_base_train_r = '/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/Train_r/'
dir_base_test_r = '/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/Test_r/'

path_in = "/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/Train/"
path_out = "/Users/sdvlop/Documents/UFPR/2sem2017/AprendizadodemaquinaCI171/trabalho/simpsons/Train_out/"



def nb_classif(X_train, X_test, y_train, y_test):
    X_train = X_train.toarray()

    gnb = BernoulliNB()
    print ('Fitting NB')
    gnb.fit(X_train, y_train)

    print ('Predicting...')
    y_pred = gnb.predict(X_test)

    # mostra o resultado do classificador na base de teste
    print (gnb.score(X_test, y_test))

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print (cm)


    print('Probabilities...')
    probs = gnb.predict_proba(X_test)

    i = 0
    c_probabilidade_alta = 0
    c_probabilidade_baixa = 0
    e_probabilidade_alta = 0
    e_probabilidade_baixa = 0
    while i < y_test.shape[0]:
        #		print ('Dado', X_test[0,i])
        classe_correta = y_test[i]
        #		print('Classe correta:', classe_correta)
        classe_escolhida = np.argmax(probs[i])
#        print('Classe escolhida:', classe_escolhida)
        maior_probabilidade = probs[i, classe_escolhida]
#        print('Maior Probabilidade:', maior_probabilidade)
        if classe_correta == classe_escolhida:
            if maior_probabilidade >= 0.8:
                c_probabilidade_alta += 1
            else:
                c_probabilidade_baixa += 1
        else:
            if maior_probabilidade >= 0.8:
                e_probabilidade_alta += 1
            else:
                e_probabilidade_baixa += 1
        i += 1
    print('Classe correta com probabilidade >= 0.8:', c_probabilidade_alta)
    print('Classe correta com probabilidade < 0.8:', c_probabilidade_baixa)
    print('Classe errada com probabilidade >= 0.8:', e_probabilidade_alta)
    print('Classe errada com probabilidade < 0.8:', e_probabilidade_baixa)
    return()

###############################


def dt_classif(X_train, X_test, y_train, y_test):

    clf = tree.DecisionTreeClassifier()

    print ('Fitting DT')
    clf.fit(X_train, y_train)

    # predicao do classificador
    print ('Predicting...')
    y_pred = clf.predict(X_test)

    # mostra o resultado do classificador na base de teste
    print (clf.score(X_test, y_test))

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print (cm)

    return()

def knn_classif(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)

    print ('Fitting Knn')
    clf.fit(X_train, y_train)

    # predicao do classificador
    print ('Predicting...')
    y_pred = clf.predict(X_test)

    # mostra o resultado do classificador na base de teste
    print (clf.score(X_test, y_test))

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print (cm)

    return()



def features_img(local_img):

#    local_img = image.load_img(img_path, target_size=(200, 200))
    x = image.img_to_array(local_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x) # extrai características

#    print()
#    print(features.shape)
#    print(features[0][0][0].shape)
    # print(features)
    reshaped_features = features.reshape(1, 512 * 6 * 6)
#    print(reshaped_features)
#    print(reshaped_features.shape)
    return(reshaped_features)

def image2svm(base, nome): # cria arquivo csv para entrada

    list = os.listdir(dir_base + base)  # dir is your directory path
    number_files = len(list)
    print (number_files)
    dataset_list = [[] for m in range(number_files)]
#    print(dir_base+nome)
    k = 0
    with open(dir_base + nome, 'w') as csvfile:
        dataset_writer = csv.writer(csvfile, delimiter=',',
                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for filename in glob.glob(dir_base + base + '/' + '*.jpg'):  # lendo todas as imagens, assuming jpg
#            print('filename', filename)
#            img1 = data.imread(filename, 0)
            img1 = image.load_img(filename, target_size=(200, 200))


#            img2 = data.load(img1)

#            i = img1.shape[0] - 0
#            j = img1.shape[1] - 0
            classe = -1
#            print (i, j)
            img2 = img1 # [0:i, 0:j]  # segmento da imagem
#            plt.imshow(img2)
#            plt.show()
            bart = re.search('bart', filename)
            if bart:
                classe = 0
                #        data.append(classe)
#                print ('filename:', filename)
            homer = re.search('homer', filename)
            if homer:
                classe = 1
            lisa = re.search('lisa', filename)
            if lisa:
                classe = 2
            maggie = re.search('maggie', filename)
            if maggie:
                classe = 3
            marge = re.search('marge', filename)
            if marge:
                classe = 4

            csv_classe = (classe)
            dataset_list[k].append(csv_classe)

            features_f = features_img(img2).flatten()
            features_l = features_f.tolist()
            dataset_list[k].extend(features_l)
#            print (dataset_list[k])
            dataset_writer.writerow(dataset_list[k]) # arquivo csv
            k += 1
    return ()


if __name__ == '__main__':

    nome_csv_train = 'csv_train_r'
    nome_csv_test = 'csv_test_r'
    nome_csv_valid = 'csv_valid_r'
    simpsons = '/Users/sdvlop/PycharmProjects/AM/simpsons/simpsons/'

    image2svm('train_r', nome_csv_train)
    os.system(
        simpsons + "csv2libsvm.py " + dir_base + nome_csv_train + ' ' + dir_base + "libsvm_train.data 0")

    image2svm('test_r', nome_csv_test)
    os.system(
        simpsons + "csv2libsvm.py " + dir_base + nome_csv_test + ' ' + dir_base + "libsvm_test.data 0")

    image2svm('valid_r', nome_csv_valid)
    os.system(
        simpsons + "csv2libsvm.py " + dir_base + nome_csv_valid + ' ' + dir_base + "libsvm_valid.data 0")



    print ("Loading train...")
    X_train, y_train = load_svmlight_file(dir_base + 'libsvm_train.data')




    clf = RandomForestClassifier()   # classificador para extrair as melhores características
    print('X_train.shape: ', X_train.shape)

    clf = clf.fit(X_train,y_train )
#    print(clf.feature_importances_)
#    with open(dir_base + 'im_features', 'w') as csvfile:
#        dataset_writer = csv.writer(csvfile, delimiter=',',
#                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#        dataset_writer.writerow(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    X_train_r = model.transform(X_train)
    print('X_train_r.shape: ', X_train_r.shape)


    print ("Loading test...")
    X_test, y_test = load_svmlight_file(dir_base + 'libsvm_test.data')

#    clf = RandomForestClassifier()
    print('X_test_r.shape: ',X_test.shape)

    clf = clf.fit(X_test,y_test )
#    print(clf.feature_importances_)
#    with open(dir_base + 'im_features', 'w') as csvfile:
#        dataset_writer = csv.writer(csvfile, delimiter=',',
#                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#        dataset_writer.writerow(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    X_test_r = model.transform(X_test)
    print('X_test_r.shape: ', X_test_r.shape)


    print ("Loading valid...")
    X_valid, y_valid = load_svmlight_file(dir_base + 'libsvm_valid.data')
    print('X_valid_r.shape: ',X_valid.shape)

    clf = clf.fit(X_valid, y_valid)
#    print(clf.feature_importances_)
#    with open(dir_base + 'im_features', 'w') as csvfile:
#        dataset_writer = csv.writer(csvfile, delimiter=',',
#                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#        dataset_writer.writerow(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    X_valid_r = model.transform(X_valid)
    print('X_valid_r.shape: ', X_valid_r.shape)



    print ("nb classifier...")
    nb_classif(X_train_r, X_test_r, y_train, y_test)
    nb_classif(X_train_r, X_valid_r, y_train, y_valid)

    print ("dt classifier...")
    dt_classif(X_train_r, X_test_r, y_train, y_test)
    dt_classif(X_train, X_valid, y_train, y_valid)

    print ("knn classifier...")
    knn_classif(X_train_r, X_test_r, y_train, y_test)
    knn_classif(X_train_r, X_valid_r, y_train, y_valid)

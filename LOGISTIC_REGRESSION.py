# obtendo DATASET de SEABORN library
# usando uma regressao logistica para determinar
# probabilidades de sobrevivencia dos tripulantes


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


Titanic_data = sns.load_dataset('titanic')
#print(Titanic_data.head())
#print(Titanic_data.count())
#seq_palette = sns.color_palette('Greys',4)
#sns.countplot(x='survived',hue='pclass',data=Titanic_data,palette=seq_palette)
#plt.show()
#print(Titanic_data.isnull().sum())
#apagando valores que possuem NaN, que atrapalha analise
Titanic_data.drop('age',axis=1,inplace=True)
Titanic_data.drop('embark_town',axis=1,inplace=True)
Titanic_data.drop('deck',axis=1,inplace=True)
Titanic_data.dropna(inplace=True)
#apagando valores que não trazem mais informações
Titanic_data.drop(['class','who','adult_male','alive','alone'],axis=1,inplace=True)
#valores categoricos sao substituidos por 0 ou 1 com pandas
sex = pd.get_dummies(Titanic_data['sex'],drop_first=True)
embarked = pd.get_dummies(Titanic_data['embarked'],drop_first=True)
#dropamos os valores e depois concatenamos os valores como inteiros
Titanic_data.drop(['sex','embarked'],axis=1,inplace=True)
Titanic_data = pd.concat([Titanic_data,sex,embarked],axis=1)
#print(Titanic_data.head())
#vamos fazer a divisao dos dados em duas features:
# X = sao predictors para indicar a sobrevivencia de pacientes,
# Y = para indicar se sobreviveu
X = Titanic_data.drop('survived',axis=1)
Y = Titanic_data['survived']
#pegando 30% do dataset para gerar o teste
Titanic_train, Titanic_test, Survived_train, Survived_test = train_test_split(X,Y,test_size =0.30
,random_state = 101)

# print('Size of the training dataset is', Titanic_train.shape[0])
# print('Size of labels of the training dataset is ', Survived_train.shape[0])
# print('Number of variables or features in the training dataset is', Titanic_train.shape[1])
# print('Size of the test data is', Titanic_test.shape[0])
# print('Size of labels of the test dataset is ', Survived_test.shape[0])
Inmodel = LogisticRegression()
Inmodel.fit(Titanic_train,Survived_train)
Survival_Predictions = Inmodel.predict(Titanic_test)
print('Model accuracy is %2.3f' %(accuracy_score(Survived_test,Survival_Predictions)))

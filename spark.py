import findspark
findspark.init()

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = "--conf spark.driver.memory=14g --conf spark.driver.maxResultSize=5g pyspark-shell"

from pyspark import SparkContext
sc = SparkContext()

filename = os.path.join(os.environ["SPARK_HOME"], 'python/pyspark/shell.py')
exec(compile(open(filename, "rb").read(), filename, 'exec'))

print(sc._conf.getAll())

## Import des différents packages nécessaires
# Traitement de données
import string
import shutil
import pandas as pd
import numpy as np

# PySpark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import split
from pyspark.sql.functions import lit
from pyspark.sql.functions import rand 

sqlContext = SQLContext(sc)

from pyspark.ml.linalg import Vectors, SparseVector, VectorUDT
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, CountVectorizerModel

from pyspark.ml import Model
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier 
from pyspark.ml.classification import NaiveBayes

from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel 
from pyspark.ml.classification import NaiveBayesModel


dir = '/Users/Mathieu/Documents/Centrale/Data Science/Projet'
os.chdir(dir)


# Fonction import des datas
train_cv = 1 # 1 for training / 0 to load the model
cvModelPath = './Data/count-vectorizer-model'

def loadData(data_ingestion, train_cv = 1, binarize = True, minDF = 3, TFIDF_b = False, PCA_b = False, PCA_k = 1000):
    if train_cv:
        # we train cv...
        cv_model = CountVectorizer(inputCol = 'words', outputCol = 'X', minDF = minDF)
    else:
        # we load cv ! 
        cv = CountVectorizerModel.load(cvModelPath)
    
    tokenizer = Tokenizer(inputCol = "comment", outputCol = "words")
    
    # Creation of an empty DataFrame
    field1 = StructField('score',IntegerType(),True)
    field2 = StructField('X',VectorUDT() ,True)
    
    fields = []
    fields.append(field1)
    fields.append(field2)
    
    schema = StructType(fields)
    
    X = spark.createDataFrame(sc.emptyRDD(), schema)
    
    # Ingestion par fichier
    for filePath in data_ingestion:
        file = sc.textFile(filePath)
        data = file.map(lambda line: line.split("\t")).toDF()
        data = data.withColumnRenamed('_2', 'comment')
        
        data = data.withColumnRenamed('_1', 'score') 
        data = data.withColumn('score', data['score'].cast(IntegerType()))
        
        data = tokenizer.transform(data)
        
        if train_cv :
            cv = cv_model.fit(data)
            
        data = cv.transform(data)
        
        X = X.union(data.select('score', 'X'))
    
    try : 
        shutil.rmtree(cvModelPath, ignore_errors = True)
    except:
        pass
    
    cv.save(cvModelPath)
    
    if binarize:
        X_1 = X.where((X.score == 4)|(X.score == 5)).withColumn('score', lit(1))
        X_0 = X.where((X.score == 0) | (X.score == 1)| (X.score == 2) | (X.score == 3)).withColumn('score', lit(0))
        X = X_1.union(X_0)
     
    if TFIDF_b:
        idf = IDF(inputCol="X", outputCol="X_TFIDF")
        model = idf.fit(X)
        X = model.transform(X)
        X = X.select('score', 'X_TFIDF')
        X = X.withColumnRenamed('X_TFIDF', 'X')
        
    if PCA_b:
        pca = PCA(k=PCA_k, inputCol="X", outputCol="X_PCA")
        model = pca.fit(X)
        X = X.select('score', 'X_PCA')
        X = X.withColumnRenamed('X_PCA', 'X')
        
    return(X)


 # Evaluation des modèles
from beautifultable import BeautifulTable

def evaluate(df, confusion = True, labelCol = 'score', predictionCol = 'prediction'):
    tp = df.select(labelCol, predictionCol).where(labelCol +'== 1').where(predictionCol +'== 1').count()
    tn = df.select(labelCol, predictionCol).where(labelCol +'== 0').where(predictionCol +'== 0').count()
    fp = df.select(labelCol, predictionCol).where(labelCol +'== 0').where(predictionCol +'== 1').count()
    fn = df.select(labelCol, predictionCol).where(labelCol +'== 1').where(predictionCol +'== 0').count()
    tot = df.count()

    table = BeautifulTable()
    if confusion:
        table.append_row(["Prediction \ Score", "Positive", "Negative"])
        table.append_row(["1", tp, fp])
        table.append_row(["0", fn, tn])
        table.append_row(["Accuracy", "(tp+tn)/N", (tp+tn)/tot])
    else:
        table.append_row(["Metrics", "Formula", "Value"])
        table.append_row(["Accuracy", 'TP+TN/(P+N)', (tp+tn)/tot])
        table.append_row(["Balanced accuracy", '(TP/P+TN/N)/2', (tp/(tp+fp)+tn/(tn+fn))/2])
        table.append_row(["Sensitivity", 'TP/P', tp/(tp+fn)])
        table.append_row(["Specificity", 'TN/N', tn/(fp+tn)])
        table.append_row(["Positive predicitve value", 'TP/(TP+FP)', tp/(tp+fp)])
        table.append_row(["Negative predicitve value", 'TN/(TN+FN)', tn/(tn+fn)])
        table.append_row(["False positive rate", 'FP/N', fp/(fp+tn)])
        table.append_row(["False negative rate", 'FN/P', fn/(fn+tp)])

    print(table)


# Creation d'un classifieurs à base de vote pour un ensemble de classifieurs
class VoteClassifier(Model):
    def __init__(self, *models):
        self._classifiers = models
        
    def transform_vote(self, X, featureCol = 'X', labelCol = 'score'):
        df = X.select(featureCol, labelCol)
    
        fields = []
        for c in self._classifiers:
            # List of all column containing a classifier prediction
            field_name = 'prediction_' + str(len(fields)+1)
            fields.append(field_name)
            
            # Apply and store the classification result of classifier c
            Y = c.transform(X)
            Y = Y.withColumnRenamed(featureCol, 'feats')
            df = df.join(Y.select('prediction', 'feats'), df[featureCol] == Y.feats).drop('feats')
            df = df.withColumnRenamed('prediction', 'prediction_' + str(len(fields)))
        
        # Compute the majority vote
        df = df.withColumn('confidence_vote', sum(df[col] for col in fields)/len(fields))
        df = df.withColumn('prediction_vote', f.when(df.confidence_vote > 0.5, 1.0).otherwise(0.0))
        df = df.withColumn('confidence_vote', f.when(df.prediction_vote == 0.0, 1 - df.prediction_vote).otherwise(df.prediction_vote))
        return(df)


# Import des data
training_large = [ dir + '/Data/stemmed_amazon_500k_train.txt']
test_large = ['./Data/stemmed_amazon_500k_test.txt']
test_imbd = [dir + '/Data/imdb_yelp.txt']

X_train_large = loadData(training_large, minDF = 1, TFIDF_b = True)
X_train_large = X_train_large.orderBy(rand())
X_test_large = loadData(test_large, train_cv = 0, TFIDF_b = True)
X_test_imbd = loadData(test_imbd, train_cv = 0, TFIDF_b = True)


X_train_large.groupby('score').count().show()
X_test_large.groupby('score').count().show()
X_test_imbd.groupby('score').count().show()


# Model path
NB_model_path = './Model/NB_model_500k'
LR_model_path = './Model/LR_model_500k'
RF_model_path = './Model/RF_model_500k'


# Naive Bayes
NB = NaiveBayes(modelType = "multinomial", labelCol = "score", featuresCol = "X")
NB_model = NB.fit(X_train_large)
NB_model.save(NB_model_path)


# Logistic Regression
LR = LogisticRegression(elasticNetParam = 0, regParam = 6.969697, labelCol = 'score', featuresCol = 'X')
LR_model = LR.fit(X_train_large)
LR_model.save(LR_model_path)


# Random Forest
RF = RandomForestClassifier(numTrees = 100, maxDepth = 15, labelCol = "score", featuresCol = "X")
RF_model = RF.fit(X_train_large)
RF_model.save(RF_model_path)

# Loading all trained models
NB_Model = NaiveBayesModel.load(NB_model_path)
LR_Model = LogisticRegressionModel.load(LR_model_path)
RF_Model = RandomForestClassificationModel.load(RF_model_path)

voteClassifier = VoteClassifier(NB_Model, LR_Model, RF_Model)
evaluate(voteClassifier.transform_vote(X_test_large), confusion = False, predictionCol = 'prediction_vote')
evaluate(voteClassifier.transform_vote(X_test_imbd), confusion = False, predictionCol = 'prediction_vote')
voteClassifier.transform_vote(X_test_imbd).show()

# Accuracy: (TP+TN)/N
# Positive Predicitve Value: TP/(TP+FP)
# Negative Predicitve Value: TN/(TN+FN)
import matplotlib.pyplot as plt

accuracy = [0.694, 0.831, 0.771]
accuracy_transfer = [0.709, 0.554, 0.532]
NPV = [0.369, 0.591, 0.879]
NPV_transfer = [0.697, 0.928, 0.958]
PPV = [0.897, 0.532, 0.453]
PPV_transfer = [0.723, 0.928, 0.52]

ind = 3*np.arange(len(accuracy))
ind2 = np.linspace(start = 0, stop = max(ind)+1, num = 100)

acc_vote = 0.828
acc = acc_vote*np.ones(len(ind2))
acc_transfer_vote = 0.449
acc_transfer = acc_transfer_vote*np.ones(len(ind2))

npv_vote = 0.537
npv = npv_vote*np.ones(len(ind2))
npv_transfer_vote = 0.932
npv_transfer = npv_transfer_vote*np.ones(len(ind2))

ppv_vote =  0.874
ppv = ppv_vote*np.ones(len(ind2))
ppv_transfer_vote = 0.431
ppv_transfer = ppv_transfer_vote*np.ones(len(ind2))

names = ('NB', 'LR', 'RF')

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches((18,3))

plt.setp(axes, xticks=ind+0.5, xticklabels=names)

plt.sca(axes[0])
plt.bar(ind, np.array(accuracy), label = 'Accuracy', width = 0.75)
plt.bar(ind+1, np.array(accuracy_transfer), label = 'Précision en Transfer Learning', width = 0.75)
plt.plot(ind2, acc_transfer, label = 'Vote en Transfer Learning')
plt.plot(ind2, acc, label = 'Vote')
plt.ylim(0,1)
plt.title('Accuracy', fontsize=16)

plt.sca(axes[1])
plt.bar(ind, np.array(NPV), label = 'Métrique', width = 0.75)
plt.bar(ind+1, np.array(NPV_transfer), label = 'Métrique en Transfer Learning', width = 0.75)
plt.plot(ind2, npv_transfer, label = 'Vote en Transfer Learning')
plt.plot(ind2, npv, label = 'Vote')
plt.ylim(0,1)
plt.title('Negative Predictive Value', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=4)

plt.sca(axes[2])
plt.bar(ind, np.array(PPV), label = 'Accuracy', width = 0.75)
plt.bar(ind+1, np.array(PPV_transfer), label = 'Précision en Transfer Learning', width = 0.75)
plt.plot(ind2, ppv_transfer, label = 'Vote en Transfer Learning')
plt.plot(ind2, ppv, label = 'Vote')
plt.ylim(0,1)
plt.title('Positive Predictive Value', fontsize=16)

plt.show()
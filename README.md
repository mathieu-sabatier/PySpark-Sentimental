# PySpark-Sentimental
A Sentimental Analysis Student Project carried out with Spark

The 'spark.py' file includes all functions and code needed for a **simple** sentimental analysis classifier. 

*loadData* ingests data from the files given by the elements of **data_ingestion** list and return a RDD.
Optionnaly, TF-IDF and/or PCA are optionnaly ran. 

*evaluate* prints the confusion matrix of a binary classifier OR all standards metrics using beautifultable package

*VoteClassifier* is a class allowing for a classifier based on votes leveraging PySpark API. 

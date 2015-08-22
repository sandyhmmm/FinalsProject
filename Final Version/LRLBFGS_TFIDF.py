from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from BeautifulSoup import BeautifulSoup
from nltk.corpus import stopwords
import re

def review_to_words(raw_review):
    # Function tto clean the reviews and resturn words from paragraphs of reviews
    #
    # removing html tags using the beautiful soup api
    review_text = BeautifulSoup(raw_review).text
    #
    # Replacing non alphabetic characters with space       
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # converting to consistant lower cases
    words = letters_only.lower().split()                                            
    # 
    # removing stop words from the reviews
    meaningful_words =  [w for w in words if not w in stops]   
    #
    # return as paragraph
    return " ".join( meaningful_words)   

#instantiating stop words from nltk
stops = set(stopwords.words("english")) 
#fetching training data from S3
lines = sc.textFile("s3://spark-project-data/labeledTrainData.tsv")
#removing header
rows = lines.zipWithIndex().filter(lambda (row,index): index > 0).keys()
#getting the columns separated by tabs
parts = rows.map(lambda l: l.split("\t"))
#creating rows rdd
review = parts.map(lambda p: Row(id=p[0], label=float(p[1]), 
    review=review_to_words(p[2])))
#creating Dataframe
schemeReview = sqlContext.createDataFrame(review)
#tokenizing the paragraphs for words
tokenizer = Tokenizer(inputCol="review", outputCol="words")
#transformation
wordsData = tokenizer.transform(schemeReview)
#Hashing the words input
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=300)
#transforming the data to hash
featurizedData = hashingTF.transform(wordsData)
#instantiating the IDF model
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
selectData = rescaledData.select("label","features","id")
#Creating RDD of LabeledPoints
lpSelectData = selectData.map(lambda x : (x.id, LabeledPoint(x.label,x.features)))
#Spliting the data for training and test
(trainingData, testData) = lpSelectData.randomSplit([0.9, 0.1])
# training the Logistic regression with LBFGS model
lrm = LogisticRegressionWithLBFGS.train(trainingData.map(lambda x: x[1]), iterations=10)
#fetching the labels and predictions for test data
labelsAndPreds = testData.map(lambda p: (p[0],p[1].label, lrm.predict(p[1].features)))
#calculating the accuracy and printing it.
accuracy = labelsAndPreds.filter(lambda (i, v, p): v == p).count() / float(testData.count())
print("Accuracy = " + str(accuracy))















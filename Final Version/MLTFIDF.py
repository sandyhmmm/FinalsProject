from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import org.apache.spark.mllib.linalg.Vectors
from array import array

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).text
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                                            
    # 
    # 4. Remove stop words
    meaningful_words =  [w for w in words if not w in stops]   
    #
    # 5. Join the words back into one string separated by space, 
    # and return the result.
    return " ".join( meaningful_words)   

stops = set(stopwords.words("english")) 
lines = sc.textFile("s3://spark-project-data/unlabeledTrainData.tsv")
rows = lines.zipWithIndex().filter(lambda (row,index): index > 0).keys()
parts = rows.map(lambda l: l.split("\t"))

review = parts.map(lambda p: Row(id=p[0], label=float(p[1]), 
	review=review_to_words(p[2])))


schemeReview = sqlContext.createDataFrame(review)
(trainingData, testData) = schemeReview.randomSplit([0.6, 0.4])
tokenizer = Tokenizer(inputCol="review", outputCol="words")
wordsData = tokenizer.transform(schemeReview)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=300)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
selectData = rescaledData.select("label","features")

#(trainingData, testData) = selectData.randomSplit([0.6, 0.4])
lr = LogisticRegression(maxIter=5, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF,idf, lr])
model = pipeline.fit(trainingData)
selected = model.transform(testData).select('review', 'label', 'prediction')

# Build a parameter grid.
paramGrid = ParamGridBuilder().addGrid(hashingTF.numFeatures, [300, 400]).addGrid(lr.regParam, [0.01, 0.1, 1.0]).build()

#Set up cross-validation.
cv = CrossValidator().setNumFolds(3).setEstimator(pipeline).setEstimatorParamMaps(paramGrid).setEvaluator(BinaryClassificationEvaluator())

#Fit a model with cross-validation.
cvModel = cv.fit(trainingData)



testTransform = cvModel.transform(testData)

predictions = testTransform.select("review", "prediction", "label")

predictionsAndLabels = predictions.map(lambda x : (x[1], x[2]))

trainErr = predictionsAndLabels.filter(lambda r : r[0] != r[1]).count() / float(testData.count())

print("TrainErr: "+str(trainErr))
  

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(testTransform, {evaluator.metricName: "areaUnderPR"})
evaluator.evaluate(testTransform, {evaluator.metricName: "areaUnderROC"})







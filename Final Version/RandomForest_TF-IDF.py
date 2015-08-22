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
lines = sc.textFile("s3://spark-project-data/labeledTrainData.tsv")
rows = lines.zipWithIndex().filter(lambda (row,index): index > 0).keys()
parts = rows.map(lambda l: l.split("\t"))

review = parts.map(lambda p: Row(id=p[0], label=float(p[1]), 
	review=review_to_words(p[2])))
schemeReview = sqlContext.createDataFrame(review)
tokenizer = Tokenizer(inputCol="review", outputCol="words")
wordsData = tokenizer.transform(schemeReview)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=300)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
selectData = rescaledData.select("label","features")
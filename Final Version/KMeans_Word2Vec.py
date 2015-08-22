from pyspark.sql import SQLContext, Row
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
from BeautifulSoup import BeautifulSoup
from nltk.corpus import stopwords
from pyspark.ml.feature import Word2Vec
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import StringIndexer
import nltk.data
import re


def paragraph_to_wordlist( raw_review):
    # Function to clean data
    #
    # removing html tags using BeautifulSoup api
    review_text = BeautifulSoup(raw_review).text
    #  
    # removing non-alpahbetical data
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # converting to consistant lowercase
    words = review_text.lower().split()
    return(words)

def paragraph_to_sentences(review,tokenizer):
    # Function to clean data, to create sentences from paragraphs of reviews.
    #
    # Use NLTK tokenizer to form sentences from the paragraph reviews
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # Loop over each sentence in the paragraph
    sentences = []
    for raw_sentence in raw_sentences:
        # Skipping empty sentances
        if len(raw_sentence) > 0:
            # clean sentences using paragraph_to_wordlist 
            sentences.append(paragraph_to_wordlist(raw_sentence))
    return sentences

#instantiating tokenizer for splitting sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#getting the set of stopwords
stops = set(stopwords.words("english")) 

#instantiating the list for sentences
new_sentence = []

#Training the Word2Vec model
#Fetching the unlabelled data from S3.
u_lines = sc.textFile("s3://spark-project-data/unlabeledTrainData.tsv")
#removing the header
u_rows = u_lines.zipWithIndex().filter(lambda (row,index): index > 0).keys()
#getting values of each column(spliting by tab)
u_parts = u_rows.map(lambda l: l.split("\t"))
#Creating a RDD of Rows contining list of lines of each review and collecting it as a list
u_review = u_parts.map(lambda p: paragraph_to_sentences(p[1],tokenizer)).collect()
#Joining the list together to form a single list
for review in u_review:
    new_sentence += review

#Creating a RDD of sentences from the list
u_sentance = sc.parallelize(new_sentence)
#Creaing a RDDD of rows
u_sentenceDF = u_sentance.map(lambda s: Row(sentence=s))
#Converting it to a dataframe
sentenceDF = sqlContext.createDataFrame(u_sentenceDF,["sentence"])
#instantiating the word2Vec Model
word2vec = Word2Vec()
#Training the model for vectorsize 300
wvModel = Word2Vec(vectorSize=300,minCount=40 ,seed=42, inputCol="sentence", outputCol="features").fit(sentenceDF)

#Training the Classification algorithm
#Fetching the data from S3
lines = sc.textFile("s3://spark-project-data/labeledTrainData.tsv")
#Removing the header
rows = lines.zipWithIndex().filter(lambda (row,index): index > 0).keys()
#Getting the columns
parts = rows.map(lambda l: l.split("\t"))
#creating RDD of reviews
review = parts.map(lambda p: Row(id=p[0], label=float(p[1]), 
	sentence=paragraph_to_wordlist(p[2])))

#creating the dataframe
reviewDF = sqlContext.createDataFrame(review)
#transforming the words to vectors using the trained model
transformDF = wvModel.transform(reviewDF)
#segregating the labels and features
selectData = transformDF.select("label","features","id")

selectRDD = selectData.map(lambda s: s.features)
(trainingData, testData) = selectRDD.randomSplit([0.6, 0.4])
clusters = KMeans.train(trainingData, 2, maxIterations=10,
        runs=10, initializationMode="random")

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = trainingData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

#initializing Streming context with a window of 10 secs
ssc = StreamingContext(sc, 10)
#fetching the input statement from S3
lines = ssc.textFileStream("s3://spark-sentimentanalysis/")
#calculating a wordcount
counts = lines.flatMap(lambda line: line.split(" "))\
             .map(lambda x: (x, 1))\
             .reduceByKey(lambda a, b: a+b)


text = lines
#creating vectors for prediction
testreview = text.map(lambda t: Row(sentence = paragraph_to_wordlist(t)))
testreviewDF = sqlContext.createDataFrame(testreview)
testVect = wvModel.transform(testreviewDF)
counts.pprint()
#predicting the sentiment
sentiment = clusters.predict(testVect.first().features)
#print it
print 'sentiment: ' + str(sentiment)
result = str(sentiment)
sc.parallelize(result).saveAsTextFile("s3://spark-sentimentanalysis/result/")

#to start and stop the ssc.
ssc.start()
ssc.awaitTermination()

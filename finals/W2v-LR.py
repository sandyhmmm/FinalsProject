from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from BeautifulSoup import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
import re
from pyspark.ml.feature import Word2Vec


def review_to_wordlist( raw_review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).text
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

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

def review_to_sentences(review,tokenizer,remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    return sentences

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(stopwords.words("english")) 

new_sentence = []
u_lines = sc.textFile("s3://spark-project-data/unlabeledTrainData.tsv")
u_rows = u_lines.zipWithIndex().filter(lambda (row,index): index > 0).keys()
u_parts = u_rows.map(lambda l: l.split("\t"))
u_review = u_parts.map(lambda p: review_to_sentences(p[1],tokenizer)).collect()
for review in u_review:
    new_sentence += review


u_sentance = sc.parallelize(new_sentence)
r_sentenceDF = u_sentance.map(lambda s: Row(sentence=s))
sentenceDF = sqlContext.createDataFrame(r_sentenceDF,["sentence"])
word2vec = Word2Vec()
model = Word2Vec(vectorSize=300,minCount=40 ,seed=42, inputCol="sentence", outputCol="features").fit(sentenceDF)


lines = sc.textFile("s3://spark-project-data/labeledTrainData.tsv")
rows = lines.zipWithIndex().filter(lambda (row,index): index > 0).keys()
parts = rows.map(lambda l: l.split("\t"))
review = parts.map(lambda p: Row(id=p[0], label=float(p[1]), 
	sentence=review_to_wordlist(p[2])))
reviewDF = sqlContext.createDataFrame(review)
transformDF = model.transform(reviewDF)

selectData = transformDF.select("label","features")
(trainingData, testData) = selectData.randomSplit([0.6, 0.4])
lr = LogisticRegression(maxIter=5, regParam=0.01)
model = lr.fit(trainingData)
result =  model.transform(testData)

u_lines.unpersist()
u_rows.unpersist()
u_parts.unpersist()
u_review.unpersist()

lines.unpersist()
rows.unpersist()


evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(result, {evaluator.metricName: "areaUnderROC"})
evaluator.evaluate(result, {evaluator.metricName: "areaUnderPR"})



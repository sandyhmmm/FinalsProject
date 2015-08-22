import re
from bs4 import BeautifulSoup
from operator import add
import plotly.plotly as py
from plotly.graph_objs import *
from nltk.corpus import stopwords


file = sc.textFile("s3://spark-project-data/unlabeledTrainData.tsv")

header = file.first()



data = file.filter(lambda x : x != header).map(lambda x : x.split('\t'))

stops = set(stopwords.words("english"))

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




reviews = data.filter(lambda x : x[1].encode('ascii','ignore')=="0").map(lambda x : x[2]).map(lambda z : review_to_words(z)).flatMap(lambda y : y.split(' '))

count = reviews.map(lambda x : (x,1)).reduceByKey(add)

sortedwords = count.map(lambda (a, b): (b, a)).sortByKey(0, 1).map(lambda (a, b): (b, a))


X = sortedwords.map(lambda x : x[0]).take(100)
Y = sortedwords.map(lambda x: x[1]).take(100)

trace = Bar(x = X, y = Y)

data = Data([trace])
py.plot(data)



q = data.filter(lambda x: x[1].encode('ascii','ignore') == "0")
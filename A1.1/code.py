from sklearn.feature_extraction.text import CountVectorizer  # l
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC  
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import defaultdict

import cPickle


DefaultTokenizer = TfidfVectorizer().build_tokenizer()

STOPLIST = set(stopwords.words('english'))

######### ########## ######### CREATE SENTIMENT DICTIONARY ########## ######### ###########

f = open("sentiment_words.txt", "r")
lines = f.readlines()
sentiment_dict = defaultdict(int)
for line in lines:
    parsed = [tok.strip() for tok in line.split("\t")]
    sentiment_dict[parsed[0]] = int(parsed[1])
f.close()

######### ########## ######### ########## ######### ########## ######### ########## #######


def clean_negate(words, i):
    sz = len(words)
    i+=1
    j = 0
    while i < sz and not words[i] in SYMBOLS and j < 5:
        words[i] = "NEG_" + words[i]
        i+=1
        j+=1
    return i


def clean_words(words):
    i = 0
    while (i < len(words)):
       # if words[i] in ["not", "nott" ,"nnott", "no", "noo", "never", "neverr"]:
        if re.match(neg_match, words[i]):
            i = clean_negate(words, i)
        else:
            i+=1


class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def regexClean(token):
	y = re.sub(r'(.)\1{2,}', r'\1\1\1' ,token)
	z = re.sub(r'-{2,}', r'', y)
	return z

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
	text = text.lower()
	text = text.strip().replace("\n", " ").replace("\r", " ")
	mentionFinder = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)", re.IGNORECASE)
	text = mentionFinder.sub("", text)
	text = regexClean(text) 

	not_destroyer = re.compile(r"\b(no{1,3}t{1,3}|can't|won't|no{1,3}|never{1,3}|haven't|hasn't|cannot|don't|isn't|ain't)\b")
	text = not_destroyer.sub("NOT", text)

	urlFinder = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.IGNORECASE)
	text = urlFinder.sub("url", text)
		    
	text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
	return text


class MagicFeaturesA(TransformerMixin):

	def fit(self, *args): return self

	def transform(self, X):
		def magicfn(text):
			if "lol" in text or "xD" in text:
				return 1
			elif "!" in text:
				return 2
			else: return 0

		return np.array([magicfn(text) for text in X]).reshape(-1,1)

class MagicFeaturesB(TransformerMixin):
    def transform(self, X):
        def score(sample):
            tokens = DefaultTokenizer(sample)
            scores = 0
            for tok in tokens:
                scores += sentiment_dict[tok]
            return scores

        return np.array([score(sample) for sample in X]).reshape(-1,1)

    def fit(self, *args):
        return self


FeatureSet = FeatureUnion([
			('new_features' ,Pipeline([('f1', MagicFeaturesA()), ('normalize', OneHotEncoder()) ]) ),
			('f2', TfidfVectorizer(binary = True, ngram_range = (1,2)) ) ,
			('f3', MagicFeaturesB())
	])

clf = LogisticRegression()
pipe = Pipeline( [ ('cleanText', CleanTextTransformer()),('features', FeatureSet)  ,  ('logistic regression', clf)  ]   )


print "Pipeline created!"
f = open('new_train_utf.csv', 'r')
lines = f.readlines()
train = 1440000
test = train
total = 1600000


X = []
Y = []
for i in xrange(total):
    X.append(lines[i][5:-2])
    Y.append(int(lines[i][1]))
    
print "Data extracted!"

f.close()

pipe.fit(X, Y)



print "Training Done!"


save_file = open("model.cPickle",'wb')
cPickle.dump(pipe,save_file,-1)
save_file.close()


testData = [line[5:-2] for line in lines[test:total]]
testLabels = [int(line[1]) for line in lines[test:total]]

preds = pipe.predict(testData)
print("accuracy:", accuracy_score(testLabels, preds))  





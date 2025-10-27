from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load model
loaded_model = None
with open('model/basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

# Load vectorizer
vectorizer = None
with open('model/count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)

# Making a prediction
# Output will be 'FAKE' or 'REAL'
prediction = loaded_model.predict(vectorizer.transform(['This is fake news']))[0]

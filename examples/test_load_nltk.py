from nltk.tokenize import PunktTokenizer
import nltk
import nltk.data

nltk.download('punkt_tab')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

# nltk.download()

# tokenizer = PunktTokenizer()
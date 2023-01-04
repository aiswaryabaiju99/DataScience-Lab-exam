import nltk
#nltk.download()
from nltk.util import ngrams
sample="amal jyothi college of engineering kanjirappaly kottayam"
NGRAMS = ngrams(sequence=nltk.word_tokenize(sample),n=3)
for grams in NGRAMS:
    print(grams)
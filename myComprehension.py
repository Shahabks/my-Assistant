from __future__ import print_function
import sys
def my_except_hook(exctype, value, traceback):
        print('There has been an error in the system')
sys.excepthook = my_except_hook
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from textatistic import Textatistic
import readability
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np # linear algebra
from nltk.sentiment import SentimentAnalyzer
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify.util import apply_features, accuracy as eval_accuracy
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import (
    BigramAssocMeasures,
    precision as eval_precision,
    recall as eval_recall,
    f_measure as eval_f_measure,
)
from collections import defaultdict
import spacy
from spacy import displacy
from IPython.display import display, HTML
from collections import Counter
import textacy.extract
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
from textblob import Word
from textblob.wordnet import VERB
import pandas as pd
import re
import string
import os
from nltk.corpus import cmudict
from langdetect import detect 
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
import codecs
import urllib.request
from urllib.request import urlopen
import sys

try:
    def url_to_string(url):
        res = requests.get(url)
        html = res.text
        soup = BeautifulSoup(html, 'html5lib')
        for script in soup(["script", "style", 'aside']):
            script.extract()
        return " ".join(re.split(r'[\n\t]+', soup.get_text()))

    def url2txt(url):
        res = requests.get(url)
        html = res.text
        soup = BeautifulSoup(page.content, 'html5lib.parser')
        p=list(body.children)
        return p.get_text()

    nlp = spacy.load('en_core_web_lg')

    lang=input("What is your target language? Type JP for Japanese and EN for English:    ")
    chos=input("Type UP for Uploading a text file or TY for typing your ""text"" :    ")
    if chos=="UP":
        path=input("the path to the text directory:    ")
        namae=input("the file name:   ")
        topic=str(input("please enter the topic you wish to know:    "))
        texttt=path+"/"+namae+".txt"
        #con=open(os.path.join(path,texttt))
        text = codecs.open(texttt, 'r', 'UTF-8').read()
        #text=con.read()
        #con.close()    

    else:
        text=input("please enter the text you wish to analyze:   ")
        topic=str(input("please enter the topic you wish to know:    "))

    document=text
    cxxc=detect(document)
    #if not cxxc=='en':
        #input("Please enter English texts")
        #exit()
    
    if lang=="EN":
        # Parse the text with spaCy. This runs the entire pipeline.
        doc = nlp(text)
        len(doc.ents)

        # 'doc' now contains a parsed version of text. We can use it to do anything we want!
        # For example, this will print out all the named entities that were detected:
        for entity in doc.ents:
            mnmn=(f"{entity.text} ({entity.label_})")
        def replace_name_with_placeholder(token):
            if token.ent_iob != 0 and token.ent_type_ == "PERSON":
                return "[REDACTED] "
            else:
                return token.string

        # Loop through all the entities in a document and check if they are names
        def scrub(text):
            doc = nlp(text)
            for ent in doc.ents:
                ent.merge()
            tokens = map(replace_name_with_placeholder, doc)
            return "".join(tokens)

        s = text
        #print(scrub(s))
        # Parse the document with spaCy
        doc = nlp(text)
        # Extract semi-structured statements
        statements = textacy.extract.semistructured_statements(doc, topic)
        # Print the results
        print(" ")
        print(" ")
        print("==========================")
        print("Here are the important topics in this article / the key concepts of the article:  ")
        print("----")
        for statement in statements:
            subject, verb, fact = statement
            print(f" - {fact}")
        # Load the large English NLP model
        nlp = spacy.load('en_core_web_lg')
        # The text we want to examine
        # Parse the document with spaCy
        doc = nlp(text)
        # Extract noun chunks that appear
        noun_chunks = textacy.extract.noun_chunks(doc, min_freq=3)
        # Convert noun chunks to lowercase strings
        noun_chunks = map(str, noun_chunks)
        noun_chunks = map(str.lower, noun_chunks)
        # Print out any nouns that are at least 2 words long
        for noun_chunk in set(noun_chunks):
            if len(noun_chunk.split(" ")) > 1:
                print(noun_chunk)
        #Create a list of common words to remove
        stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
                "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
                "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", 
                "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
                "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
                "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
                "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
                "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

        #Define a function to extract keywords
        def get_aspects(x):
            doc=nlp(x) ## Tokenize and extract grammatical components
            doc=[i.text for i in doc if i.text not in stop_words and i.pos_=="NOUN"] ## Remove common words and retain only nouns
            doc=list(map(lambda i: i.lower(),doc)) ## Normalize text to lower case
            doc=pd.Series(doc)
            doc=doc.value_counts().head().index.tolist() ## Get 5 most frequent nouns
            return doc
        #Apply the function to get aspects from reviews top 5, most frequent nouns, these will be the key-words/aspects
        print(" ")
        print(" ")
        print("==========================")
        print("top 5 key-words or aspects")
        print("----")
        print(get_aspects(text))
        print("==========================")
        print(" ")
        labels = [x.label_ for x in doc.ents]
        mba=Counter(labels)
        #print(mba)
        items = [x.text for x in doc.ents]
        mbd=Counter(items).most_common(8)
        print("==========================")
        print("most frequent nouns in the text")
        print(mbd)
        print(" ")
        sentences = [x for x in doc.sents]
        #displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')
        # Object of automatic summarization.
        auto_abstractor = AutoAbstractor()
        # Set tokenizer.
        auto_abstractor.tokenizable_doc = SimpleTokenizer()
        # Set delimiter for making a list of sentence.
        auto_abstractor.delimiter_list = [".", "\n"]
        # Object of abstracting and filtering document.
        abstractable_doc = TopNRankAbstractor()
        # Summarize document.
        result_dict = auto_abstractor.summarize(document, abstractable_doc)
        # Output result.
        print("==========================")
        print("Summary of the text")
        print("----")
        for sentence in result_dict["summarize_result"]:
            print(sentence)

    else:
        document=text
        # Object of automatic summarization.
        auto_abstractor = AutoAbstractor()
        # Set tokenizer for Japanese.
        auto_abstractor.tokenizable_doc = MeCabTokenizer()
        # Set delimiter for making a list of sentence.
        auto_abstractor.delimiter_list = ["。", "\n"]
        # Object of abstracting and filtering document.
        abstractable_doc = TopNRankAbstractor()
        # Summarize document.
        result_dict = auto_abstractor.summarize(document, abstractable_doc)
        # Output result.
        print("==========================")
        print("Summary of the text")
        print("----")
        for sentence in result_dict["summarize_result"]:
            print(sentence)
    print(" ")
    print("==========================")        

    s = Textatistic(document)
    print(s.counts)
    print(s.sent_count)
    levsco=s.flesch_score
    levsco= abs(levsco)
    if levsco>100:
        levsco=levsco/10
        print ("your ease of readability:   ", round(levsco))
        if levsco>=90:
            print("It sounds like a 5th grade writing")
        elif levsco>=80 and levsco<90:
            print("It sounds like a 6th grade writing")
        elif levsco>=70 and levsco<80:
            print("It sounds like a 7th grade writing")
        elif levsco>=60 and levsco<70:
            print("It sounds like a 8th or 9th grade writing")
        elif levsco>=50 and levsco<60:
            print("It sounds like a 10th to 12th grade writing")
        elif levsco>=30 and levsco<50:
            print("It sounds like a college student's writing")
        elif levsco>=0 and levsco<30:
            print("It sounds like a college graduate's writing")
        else:
            print("try again")
    else:
        print("your ease of readability:   ", round(levsco))
        if levsco>=90:
            print("It sounds like a 5th grade writing")
        elif levsco>=80 and levsco<90:
            print("It sounds like a 6th grade writing")
        elif levsco>=70 and levsco<80:
            print("It sounds like a 7th grade writing")
        elif levsco>=60 and levsco<70:
            print("It sounds like a 8th or 9th grade writing")
        elif levsco>=50 and levsco<60:
            print("It sounds like a 10th to 12th grade writing")
        elif levsco>=30 and levsco<50:
            print("It sounds like a college student's writing")
        elif levsco>=0 and levsco<30:
            print("It sounds like a college graduate's writing")
        else:
            print("try again")
    results = readability.getmeasures(document, lang='en')
    print("------------------------------------")
    #print(results['readability grades']['FleschReadingEase'])

    input("hold on!! press key to terminate the program")
except:    
    input("Please enter English texts")

class SentimentAnalyzer(object):
    """
    A Sentiment Analysis tool based on machine learning approaches.
    """

    def __init__(self, classifier=None):
        self.feat_extractors = defaultdict(list)
        self.classifier = classifier

    def all_words(self, documents, labeled=None):
       """
        Return all words/tokens from the documents (with duplicates).
        :param documents: a list of (words, label) tuples.
        :param labeled: if `True`, assume that each document is represented by a
            (words, label) tuple: (list(str), str). If `False`, each document is
            considered as being a simple list of strings: list(str).
        :rtype: list(str)
        :return: A list of all words/tokens in `documents`.
        """
       all_words = []
       if labeled is None:
           labeled = documents and isinstance(documents[0], tuple)
       if labeled == True:
           for words, sentiment in documents:
               all_words.extend(words)
       elif labeled == False:
           for words in documents:
               all_words.extend(words)
       return all_words


    def apply_features(self, documents, labeled=None):
        """
        Apply all feature extractor functions to the documents. This is a wrapper
        around `nltk.classify.util.apply_features`.

        If `labeled=False`, return featuresets as:
            [feature_func(doc) for doc in documents]
        If `labeled=True`, return featuresets as:
            [(feature_func(tok), label) for (tok, label) in toks]

        :param documents: a list of documents. `If labeled=True`, the method expects
            a list of (words, label) tuples.
        :rtype: LazyMap
        """
        return apply_features(self.extract_features, documents, labeled)


    def unigram_word_feats(self, words, top_n=None, min_freq=0):
        """
        Return most common top_n word features.

        :param words: a list of words/tokens.
        :param top_n: number of best words/tokens to use, sorted by frequency.
        :rtype: list(str)
        :return: A list of `top_n` words/tokens (with no duplicates) sorted by
            frequency.
        """
        # Stopwords are not removed
        unigram_feats_freqs = FreqDist(word for word in words)
        return [
            w
            for w, f in unigram_feats_freqs.most_common(top_n)
            if unigram_feats_freqs[w] > min_freq
        ]


    def bigram_collocation_feats(
        self, documents, top_n=None, min_freq=3, assoc_measure=BigramAssocMeasures.pmi
    ):
        """
        Return `top_n` bigram features (using `assoc_measure`).
        Note that this method is based on bigram collocations measures, and not
        on simple bigram frequency.

        :param documents: a list (or iterable) of tokens.
        :param top_n: number of best words/tokens to use, sorted by association
            measure.
        :param assoc_measure: bigram association measure to use as score function.
        :param min_freq: the minimum number of occurrencies of bigrams to take
            into consideration.

        :return: `top_n` ngrams scored by the given association measure.
        """
        finder = BigramCollocationFinder.from_documents(documents)
        finder.apply_freq_filter(min_freq)
        return finder.nbest(assoc_measure, top_n)


    def classify(self, instance):
        """
        Classify a single instance applying the features that have already been
        stored in the SentimentAnalyzer.

        :param instance: a list (or iterable) of tokens.
        :return: the classification result given by applying the classifier.
        """
        instance_feats = self.apply_features([instance], labeled=False)
        return self.classifier.classify(instance_feats[0])


    def add_feat_extractor(self, function, **kwargs):
        """
        Add a new function to extract features from a document. This function will
        be used in extract_features().
        Important: in this step our kwargs are only representing additional parameters,
        and NOT the document we have to parse. The document will always be the first
        parameter in the parameter list, and it will be added in the extract_features()
        function.

        :param function: the extractor function to add to the list of feature extractors.
        :param kwargs: additional parameters required by the `function` function.
        """
        self.feat_extractors[function].append(kwargs)


    def extract_features(self, document):
        """
        Apply extractor functions (and their parameters) to the present document.
        We pass `document` as the first parameter of the extractor functions.
        If we want to use the same extractor function multiple times, we have to
        add it to the extractors with `add_feat_extractor` using multiple sets of
        parameters (one for each call of the extractor function).

        :param document: the document that will be passed as argument to the
            feature extractor functions.
        :return: A dictionary of populated features extracted from the document.
        :rtype: dict
        """
        all_features = {}
        for extractor in self.feat_extractors:
            for param_set in self.feat_extractors[extractor]:
                feats = extractor(document, **param_set)
            all_features.update(feats)
        return all_features


    def train(self, trainer, training_set, save_classifier=None, **kwargs):
        """
        Train classifier on the training set, optionally saving the output in the
        file specified by `save_classifier`.
        Additional arguments depend on the specific trainer used. For example,
        a MaxentClassifier can use `max_iter` parameter to specify the number
        of iterations, while a NaiveBayesClassifier cannot.

        :param trainer: `train` method of a classifier.
            E.g.: NaiveBayesClassifier.train
        :param training_set: the training set to be passed as argument to the
            classifier `train` method.
        :param save_classifier: the filename of the file where the classifier
            will be stored (optional).
        :param kwargs: additional parameters that will be passed as arguments to
            the classifier `train` function.
        :return: A classifier instance trained on the training set.
        :rtype:
        """
        print("Training classifier")
        self.classifier = trainer(training_set, **kwargs)
        if save_classifier:
            save_file(self.classifier, save_classifier)

        return self.classifier


    def evaluate(
        self,
        test_set,
        classifier=None,
        accuracy=True,
        f_measure=True,
        precision=True,
        recall=True,
        verbose=False,
    ):
        """
        Evaluate and print classifier performance on the test set.

        :param test_set: A list of (tokens, label) tuples to use as gold set.
        :param classifier: a classifier instance (previously trained).
        :param accuracy: if `True`, evaluate classifier accuracy.
        :param f_measure: if `True`, evaluate classifier f_measure.
        :param precision: if `True`, evaluate classifier precision.
        :param recall: if `True`, evaluate classifier recall.
        :return: evaluation results.
        :rtype: dict(str): float
        """
        if classifier is None:
            classifier = self.classifier
        print("Evaluating {0} results...".format(type(classifier).__name__))
        metrics_results = {}
        if accuracy == True:
            accuracy_score = eval_accuracy(classifier, test_set)
            metrics_results['Accuracy'] = accuracy_score

        gold_results = defaultdict(set)
        test_results = defaultdict(set)
        labels = set()
        for i, (feats, label) in enumerate(test_set):
            labels.add(label)
            gold_results[label].add(i)
            observed = classifier.classify(feats)
            test_results[observed].add(i)

        for label in labels:
            if precision == True:
                precision_score = eval_precision(
                    gold_results[label], test_results[label]
                )
                metrics_results['Precision [{0}]'.format(label)] = precision_score
            if recall == True:
                recall_score = eval_recall(gold_results[label], test_results[label])
                metrics_results['Recall [{0}]'.format(label)] = recall_score
            if f_measure == True:
                f_measure_score = eval_f_measure(
                    gold_results[label], test_results[label]
                )
                metrics_results['F-measure [{0}]'.format(label)] = f_measure_score

        # Print evaluation results (in alphabetical order)
        if verbose == True:
            for result in sorted(metrics_results):
                print('{0}: {1}'.format(result, metrics_results[result]))

        return metrics_results

def percentage1(count, total):
   return 100 * count / total 
def percentage2(xxx):
   return 100 * xxx 
text = document
trans = document
not_punctuation = lambda w: not (len(w)==1 and (not w.isalpha()))
woallen= list(filter(not_punctuation, word_tokenize(text)))
get_word_count = lambda text: len(woallen)
get_sent_count = lambda text: len(sent_tokenize(text))
prondict = cmudict.dict()
#numsyllables_pronlist = lambda l: len(list(filter(lambda s: isdigit(s.encode('ascii', 'ignore').lower()[-1]), l)))
def numsyllables(word):
  try:
    return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
  except KeyError:
    return [0]
def text_statistics(text):
  word_count = get_word_count(text)
  sent_count = get_sent_count(text)
  s = Textatistic(text)
  syllable_count=s.sybl_count
  #moallen=list(map(lambda w: max(numsyllables(w)), word_tokenize(text)))
  #syllable_count = sum(moallen)
  return word_count, sent_count, syllable_count
flesch_formula = lambda word_count, sent_count, syllable_count : 206.835 - 1.015*word_count/sent_count - 84.6*syllable_count/word_count
def flesch(text):
  word_count, sent_count, syllable_count = text_statistics(text)
  return flesch_formula(word_count, sent_count, syllable_count)
 
fk_formula = lambda word_count, sent_count, syllable_count : 0.39 * word_count / sent_count + 11.8 * syllable_count / word_count - 15.59
def flesch_kincaid(text):
  word_count, sent_count, syllable_count = text_statistics(text)
  return fk_formula(word_count, sent_count, syllable_count)
txt=trans.split()
lent=len(word_tokenize(trans))
SentimentAnalyzer(text)
#fe=pathy+"/"+"dataset"+"/"+"essen"+"/"+"my_classifier.pickle"
#f = open(fe, 'rb')
#classifier = pickle.load(f)
qualityy=TextBlob(text)
obsub=qualityy.sentiment
if obsub[1]>0.5:
        print("The text sounds less logically structured by %:   ", "%.2f" % ((obsub[1]-0.5)*100))
else:
        print("The text sounds more logically structured by %:   ", "%.2f" % ((1-obsub[1])*100))
#f.close()
quaa=qualityy.words
def lexical_diversity(a):
    return len(set(trans))/len(trans)
v=percentage2(lexical_diversity(trans))
if v>=0.145*100 and v<0.231*100 :
        langscore="b2"
elif v>=0.231*100:
        langscore="c"
elif v>=0.121*100 and v<0.145*100:
        langscore="b1"
elif v<0.121*100 and v>=0.09*100:
        langscore="a2"
else:
        langscore="a1"  

tokenized_text=word_tokenize(text)
#print(tokenized_text)
tokenized_sent=sent_tokenize(text)
stop_words=set(stopwords.words("english"))
filtered_sent=[]
for w in tokenized_text:
        if w not in stop_words:
                filtered_sent.append(w)
#print("Filterd Sentence:",filtered_sent)
filtered_sent=filtered_sent
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
        stemmed_words.append(ps.stem(w))
#print("Stemmed Sentence:",stemmed_words)
def lexical_diversity(a):
        return len(set(a))/len(a)
def lexical_richness(a,b):
        return len(a)/len(b)
vio=percentage2(lexical_diversity(tokenized_text))
vocano=lexical_richness(filtered_sent,tokenized_text)
if vocano>=0.4 and vocano<0.5:
        langscoree="b2"
elif vocano>=0.5:
        langscoree="c"
elif vocano>=0.3 and vocano<0.4:
        langscoree="b1"
elif vocano<0.3 and vocano>=0.2:
        langscoree="a2"
else:
        langscoree="a1" 

vvv=percentage2(vocano)
#print(len(set(filtered_sent)))
#print(len(trans))
#print(len(tokenized_text))
vocab = filtered_sent
long_words = [w for w in vocab if len(w) >10]
soso=sorted(long_words)
mesu=len(soso)/len(vocab)
if mesu>0.01:
        sayy="Sofisticated words are used"
else:
        sayy=" "
text_statistics(text)
flesh=206.835-1.015*(len(tokenized_text))/2-84.6*(len(tokenized_text)*mesu*25)/(len(tokenized_text))
flesh=flesch_kincaid(text)
print(" ")
print("===========================================")

print("intelligibility level of delivery:   ", "%.2f" % (abs(flesh)))

print(" ")
print("===========================================")
#print("the lexical richness in the specific contexts %:   ",round(v,1))
print("the lexical richness in the context %:    ",round(vio,1))
print("the lexical diversity %:  ",round(vvv,1))
print(sayy)
print(" ")
fini=input("The general assessment is DONE, press any other key to terminate the programe:   ")

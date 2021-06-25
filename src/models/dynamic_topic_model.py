from gensim.models import LdaSeqModel
from gensim import corpora
from gensim.test.utils import datapath
import en_core_web_sm
import operator
from functools import reduce
import pickle
import pdb
import pyLDAvis
import numpy as np
import pandas as pd
from typing import List
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

nlp = en_core_web_sm.load()
TOPIC_NUM = 3
categories_mfd_combined = ["care.virtue", "care.vice", "fairness.virtue", "fairness.vice", "loyalty.virtue", "loyalty.vice", "authority.virtue", "authority.vice", "sanctity.virtue", "sanctity.vice", "morality.general"]
moral_types = ["harm", "fairness", "loyalty", "authority", "purity", "other"]
def get_BOW(text_data):
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    
    return corpus, dictionary

def get_documents_by_age(corpus_df: pd.DataFrame, use_stopwords: bool = True) -> dict:
    corpus_df["year"] = corpus_df["year"].apply(np.floor)
    documents_over_time = {value: [] for value in moral_types}
    for year in range(int(corpus_df["year"].max()) + 1): 
        for value in moral_types:
            utterances_year = corpus_df.loc[(corpus_df["year"] == year) & (corpus_df["type"] == value)]
            contexts_no_keywords = get_context_utterances(utterances_year, use_stopwords=use_stopwords)
            all_split = [string.split() for string in contexts_no_keywords]
            all_split = [x for x in all_split if len(x) > 0]
            documents_over_time[value].append(all_split)
    
    return documents_over_time

def get_context_utterances(utterances_df: pd.DataFrame, use_stopwords: bool = True) -> List:
    if use_stopwords:
        no_context = utterances_df.apply(remove_keywords, axis=1)
    else:
        no_context = utterances_df.apply(remove_keywords_and_stopwords, axis=1)
    return list(no_context)

def remove_keywords(row):
    return row["context"].replace(row["keywords"], "").replace("CLITIC", "")

def remove_keywords_and_stopwords(row):
    word_tokens = word_tokenize(row["context"])
    filtered_sentence = " ".join([w for w in word_tokens if not w.lower() in stop_words])
    return filtered_sentence.replace(row["keywords"], "").replace("CLITIC", "")

def fit_seqlda(articles: List) -> LdaSeqModel:
    time_slice = [len(x) for x in articles]
    corpus, id2token = get_corpus_info(articles)
    print(corpus)   
    print(id2token)
    ldaseq = LdaSeqModel(corpus=corpus, id2word = id2token,time_slice=time_slice, num_topics= TOPIC_NUM, chunksize=1)

    print(ldaseq.print_topics(time=0))

    return ldaseq

def get_corpus_info(articles: List) -> tuple:
    articles_flattened = reduce(operator.concat, articles)
    corpus, id2token = get_BOW(articles_flattened)

    return corpus, id2token


def visualize(lda_model: LdaSeqModel, corpus: List, out_filename: str, max_age: List) -> None:
    for time in range(max_age):
        out_filename_age = f"{out_filename}-{time}.html"
        doc_topic, topic_term, doc_lengths, term_frequency, vocab = lda_model.dtm_vis(time=time, corpus=corpus)
        vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths,
        vocab=vocab, term_frequency=term_frequency, sort_topics=False)
        with open(out_filename_age, "w") as out_f:
            pyLDAvis.save_html(vis_wrapper, out_f)
    #pyLDAvis.display(vis_wrapper)


#look at this source
# https://radimrehurek.com/gensim/models/ldaseqmodel.html



#a very good tool for visualization (try running it in a notebook)

""" import pyLDAvis

doc_topic, topic_term, doc_lengths, term_frequency, vocab = ldaseq.dtm_vis(time=0, corpus=corpus,)


vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths,
                               vocab=vocab, term_frequency=term_frequency, sort_topics = False)
pyLDAvis.display(vis_wrapper) """

if __name__ == "__main__":
    harm_model = pickle.load(open("./data/harm-model.p", "rb"))
    corpus_df = pickle.load(open("./data/moral_df_context_combined.p", "rb"))
    docs_by_age = get_documents_by_age(corpus_df, use_stopwords=True)
    corpus, id2token = get_corpus_info(docs_by_age["harm"])
    max_age = 6

    visualize(harm_model, corpus, "lda_vis/harm_model/age", max_age)
    assert False
    corpus_df = pickle.load(open("./data/moral_df_context_combined.p", "rb"))
    docs_by_age = get_documents_by_age(corpus_df, use_stopwords=True)
    harm_model = fit_seqlda(docs_by_age["harm"])
    with open("./data/harm-model.p", "wb") as out_f:
        pickle.dump(harm_model, out_f)

    fairness_model = fit_seqlda(docs_by_age["fairness"])
    with open("./data/fairness-model.p", "wb") as out_f:
        pickle.dump(fairness_model, out_f)
    loyalty_model = fit_seqlda(docs_by_age["loyalty"])
    with open("./data/loyalty-model.p", "wb") as out_f:
        pickle.dump(loyalty_model, out_f)
   
    authority_model = fit_seqlda(docs_by_age["authority"])
    with open("./data/authority-model.p", "wb") as out_f:
        pickle.dump(authority_model, out_f)
    purity_model = fit_seqlda(docs_by_age["purity"])
    with open("./data/purity-model.p", "wb") as out_f:
        pickle.dump(purity_model, out_f)
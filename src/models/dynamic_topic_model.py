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
TOPIC_NUM = 5
categories_mfd_combined = ["care.virtue", "care.vice", "fairness.virtue", "fairness.vice", "loyalty.virtue", "loyalty.vice", "authority.virtue", "authority.vice", "sanctity.virtue", "sanctity.vice", "morality.general"]
moral_types = ["harm", "fairness", "loyalty", "authority", "purity", "other"]
def get_BOW(text_data):
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    
    return corpus, dictionary

def get_documents_by_age(corpus_df: pd.DataFrame, use_stopwords: bool = True) -> dict:
    corpus_df["year"] = corpus_df["year"].apply(np.floor)
    documents_over_time = {value: [] for value in moral_types}
    documents_over_time_w_keywords = {value: [] for value in moral_types}
    for year in range(int(corpus_df["year"].max()) + 1): 
        for value in moral_types:
            utterances_year = corpus_df.loc[(corpus_df["year"] == year) & (corpus_df["type"] == value)]
            contexts_no_keywords = get_context_utterances(utterances_year, use_stopwords=use_stopwords)
            all_split = [string.split() for string in contexts_no_keywords]
            all_split = [x for x in all_split if len(x) > 0]
            all_with_keywords = [x.split() for x in list(utterances_year["context"])]
            documents_over_time[value].append(all_split)
            documents_over_time_w_keywords[value].append(all_with_keywords)
    
    return documents_over_time, documents_over_time_w_keywords

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
    corpus, id2token, _ = get_corpus_info(articles)
    print(corpus)   
    print(id2token)
    ldaseq = LdaSeqModel(corpus=corpus, id2word = id2token,time_slice=time_slice, num_topics= TOPIC_NUM, chunksize=1)

    print(ldaseq.print_topics(time=0))

    return ldaseq

def get_corpus_info(articles: List) -> tuple:
    articles_flattened = reduce(operator.concat, articles)
    corpus, id2token = get_BOW(articles_flattened)

    return corpus, id2token, articles_flattened


def visualize(lda_model: LdaSeqModel, corpus: List, out_filename: str, max_age: List) -> None:
    for time in range(max_age):
        out_filename_age = f"{out_filename}-{time}.html"
        doc_topic, topic_term, doc_lengths, term_frequency, vocab = lda_model.dtm_vis(time=time, corpus=corpus)
        vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths,
        vocab=vocab, term_frequency=term_frequency, sort_topics=False)
        with open(out_filename_age, "w") as out_f:
            pyLDAvis.save_html(vis_wrapper, out_f)
    #pyLDAvis.display(vis_wrapper)


def get_document_topics(corpus: List, articles_list: List, ldamodel: LdaSeqModel) -> dict:
    topics_distribution_docs = {i: [] for i in range(TOPIC_NUM)}
    for i, document in enumerate(corpus):
        topic_distribution = ldamodel.doc_topics(doc_number=i)
        for j, prob in enumerate(topic_distribution):
            document_text = articles_list[i]
            topics_distribution_docs[j].append((prob, document_text))

    for topic_num in topics_distribution_docs:
        topics_distribution_docs[topic_num] = sorted(topics_distribution_docs[topic_num], reverse=True)

    return topics_distribution_docs

def print_topics(topics_distribution: dict, out_filename: str, num_topics: int = 30) -> None:
    to_print = {topic: [] for topic in topics_distribution}
    for topic in topics_distribution:
        top_docs = [" ".join(item[1]) for item in topics_distribution[topic][:num_topics]]
        top_docs = [item.replace("CLITIC", "") for item in top_docs]
        to_print[topic] = top_docs

    with open(out_filename, "w") as out_f:
        for topic in to_print:
            out_f.write(f"=== TOPIC {topic} ===\n")
            for utterance in to_print[topic]:
                out_f.write(utterance)
                out_f.write("\n")

#look at this source
# https://radimrehurek.com/gensim/models/ldaseqmodel.html



#a very good tool for visualization (try running it in a notebook)

""" import pyLDAvis

doc_topic, topic_term, doc_lengths, term_frequency, vocab = ldaseq.dtm_vis(time=0, corpus=corpus,)


vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths,
                               vocab=vocab, term_frequency=term_frequency, sort_topics = False)
pyLDAvis.display(vis_wrapper) """

if __name__ == "__main__":
    # harm_model = pickle.load(open("./data/purity-model.p", "rb"))
    # corpus_df = pickle.load(open("./data/moral_df_context_combined.p", "rb"))
    # docs_w_keywords_flat = reduce(operator.concat, docs_by_age_w_keywords["purity"])
    # corpus, id2token, articles_list = get_corpus_info(docs_by_age["purity"])
    # max_age = 6

    # doc_topics = get_document_topics(corpus, docs_w_keywords_flat, harm_model)
    # print_topics(doc_topics, "./data/purity-sentences.txt")

    # visualize(harm_model, corpus, "lda_vis/harm_model/age", max_age)
    # assert False
    corpus_df = pickle.load(open("./data/moral_df_context_combined.p", "rb"))
    docs_by_age, docs_by_age_w_keywords = get_documents_by_age(corpus_df, use_stopwords=True)
    harm_model = fit_seqlda(docs_by_age["harm"])
    with open("./data/harm-model-5.p", "wb") as out_f:
        pickle.dump(harm_model, out_f)

    fairness_model = fit_seqlda(docs_by_age["fairness"])
    with open("./data/fairness-model-5.p", "wb") as out_f:
        pickle.dump(fairness_model, out_f)
    loyalty_model = fit_seqlda(docs_by_age["loyalty"])
    with open("./data/loyalty-model-5.p", "wb") as out_f:
        pickle.dump(loyalty_model, out_f)
   
    authority_model = fit_seqlda(docs_by_age["authority"])
    with open("./data/authority-model-5.p", "wb") as out_f:
        pickle.dump(authority_model, out_f)
    purity_model = fit_seqlda(docs_by_age["purity"])
    with open("./data/purity-model-5.p", "wb") as out_f:
        pickle.dump(purity_model, out_f)
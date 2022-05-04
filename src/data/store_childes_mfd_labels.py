import pickle
import os
import numpy as np
import pandas as pd
from itertools import product
from src.data.cluster_childes import map_categories
from src.data.clean_survey import threshold
from src.data.clustering_models import GMM

saving =  ''
certain = '_certain'
vote = 'weighted'
f_vote = True
version = 300
model_name = 'GMM'

def get_clustering_model(age,category,polarity, identity)->(bool,GMM):


    model_dir = f'data/{model_name}/{identity}/{category}-{polarity}/{age}_all_dimension_reduced.pkl'

    if not os.path.exists(model_dir):
        return False, None

    labels_utterances, probs, best_k, model = pickle.load(open(model_dir, 'rb'))

    return True, model


def get_clustering_utterances(age,category,polarity, identity)->(bool,GMM):


    model_dir = f'data/{model_name}_sentences/{identity}/{category}-{polarity}/{age}_sentences.pkl'

    if not os.path.exists(model_dir):

        return False, None

    cluster_utterances = pickle.load(open(model_dir, 'rb'))
    return True,cluster_utterances



def get_cluster_moralization(moral_df,category,sentiment, identity,cluster, vote_column = 'central vote', ):
    clusters = moral_df.loc[(moral_df.foundation == category + ' ' + sentiment) & (moral_df.identity == identity)][['cluster', vote_column]]
    cluster_moralization = list(clusters.loc[clusters.cluster == cluster][vote_column])
    if len(cluster_moralization) == 0:
        return 'Non-moral'
    print(cluster_moralization[0])
    return cluster_moralization[0]

def get_cluster_foundation(moral_df,category,sentiment, identity,cluster, vote_column = 'central foundation vote'):
    clusters = moral_df.loc[(moral_df.foundation == category + ' ' + sentiment) & (moral_df.identity == identity)][['cluster', vote_column]]
    cluster_moralization =list(clusters.loc[clusters.cluster == cluster][vote_column])
    if len(cluster_moralization) == 0:
        return ''
    return cluster_moralization[0]


def get_survey_sentence_id(sentence_df,context, category):
    sentence = sentence_df.loc[sentence_df.sentence == context]

    cluster = list(sentence['cluster'])[0]
    f_vote = list(sentence['foundation vote'])[0]
    moral_vote = list(sentence['moral vote'])[0]
    moral_vote_num = list(sentence['moral vote number'])[0]
    total_vote_num = list(sentence['total vote number'])[0]
    certainty = (moral_vote == 'Moral') and ((moral_vote_num / total_vote_num) >= threshold or f_vote == category)

    return cluster, moral_vote, f_vote, certainty

def contain_string(lst, string):

    return any([x == string for x in lst])


def read_childes():
    # childes_data_full = pickle.load(
        # open(f'./data/moral_df_context_all_moralization_version_N{version}_previous_clusters.p', 'rb'))
    childes_data_full = pickle.load(
        open(f'./data/moral_df_context_all-lemmatized.p', 'rb'))

    childes_data_full['category'] = [map_categories[c] if c in map_categories.keys() else c for c in
                                         childes_data_full['category']]


    return childes_data_full

def save_survey_age(sentence_df, moral_df):
    ages = list(range(1, 7))
    identities = ['parent', 'child']
    categories = ['harm', 'authority', 'fairness', 'purity', 'loyalty']
    sentiments = ['pos', 'neg']

    list_rows = []
    sentence_age = {}

    for age , identity, category, sentiment in list(product(ages,identities,categories,sentiments)):
        exists, cluster_utterances = get_clustering_utterances(age, category, sentiment, identity)

        if exists:
            #lets remove survey sentences
            survey_sentences = list(sentence_df.loc[(sentence_df.identity == identity) &
                                                    (sentence_df.foundation == category) & (sentence_df.sentiment == sentiment)]['sentence'])
            print(f'len survey sentence: {len(survey_sentences)}')

            for cluster, utterances in cluster_utterances.items():
                central_moralization = get_cluster_moralization(moral_df, category,sentiment,identity, cluster,vote + ' vote')
                print(age, identity, category, sentiment, cluster, central_moralization)

                for s in survey_sentences:
                    occurance = utterances.count(s)
                    if occurance > 0:
                        if s not in sentence_age:
                            sentence_age[s] = [age] * occurance
                        else:
                            sentence_age[s] += [age] * occurance

    #adding survey here
    for i, s in sentence_df.iterrows():
        category =  s['foundation']
        context = s['sentence']
        identity = s['identity']
        sentiment = s['sentiment']

        s_ages = [str(x) for x in sentence_age[context]]

        row = {'id': s['id'], 'year': ",".join(s_ages), 'identity': identity, 'foundation': category, 'sentiment': sentiment, 'context': context,
                    'foundation vote': s['foundation vote'],
                       'moral vote': s['moral vote'], 'total vote number':s['total vote number'], 'moral vote number': s['moral vote number'] }



        list_rows.append(row)


    return pd.DataFrame(list_rows)




def create_moral_df(childes_data, sentence_df, moral_df):

    non_mfd_data = childes_data.loc[childes_data.year.astype(int) < 7].loc[childes_data.category.isna()]

    mfd_df = pd.DataFrame()
    ages = list(range(1, 7))
    identities = ['parent', 'child']
    categories = ['harm', 'authority', 'fairness', 'purity', 'loyalty']
    sentiments = ['pos', 'neg']
    sentence_age = {}


    for age, identity, category, sentiment in list(product(ages, identities, categories, sentiments)):

        exists, cluster_utterances = get_clustering_utterances(age, category, sentiment, identity)


        if exists:
            #remove survey sentences
            survey_sentences = list(sentence_df.loc[(sentence_df.identity == identity) &
                                                    (sentence_df.foundation == category) & (
                                                                sentence_df.sentiment == sentiment)]['sentence'])


            for cluster, utterances in cluster_utterances.items():
                moralization = get_cluster_moralization(moral_df, category, sentiment, identity, cluster,
                                                                vote + ' vote')
                if f_vote:
                    foundation = get_cluster_foundation(moral_df, category, sentiment, identity, cluster,
                                                       vote_column=vote + ' foundation vote')
                else:
                    foundation = category


                utterances_with_no_survey = [x for x in utterances if not contain_string(survey_sentences, x)]
                for s in survey_sentences:
                    s_cluster, s_moral_vote, s_f_vote, s_certainty = get_survey_sentence_id(sentence_df, s, category)
                    if (len(certain) > 0 and s_certainty) or (len(certain) == 0):
                        occurance = utterances.count(s)
                        if occurance > 0:
                            if s not in sentence_age:
                                sentence_age[s] = [age] * occurance
                            else:
                                sentence_age[s] += [age] * occurance


                small_df = small_childes.loc[small_childes.year.astype(int) == age].loc\
                [small_childes.identity == identity].loc[small_childes.category == category].loc[small_childes.sentiment == sentiment]
                small_df = small_df.loc[small_df.context.isin(utterances_with_no_survey)]
                small_df['cluster'] = [cluster] * len(small_df)
                small_df['weighted foundation'] = [foundation] * len(small_df)
                small_df['central peripheral cluster moral relevance'] = [moralization] * len(small_df)
                small_df['from survey'] = [False] * len(small_df)
                mfd_df = mfd_df.append(small_df)


    #adding survey here
    for i, s in sentence_df.iterrows():
        category =  s['foundation']
        context = s['sentence']
        identity = s['identity']
        sentiment = s['sentiment']
        if context not in sentence_age.keys():
            continue
        # s_ages = sentence_age[context]
        s_ages = set(sentence_age[context])
        s_cluster, s_moral_vote, s_f_vote, s_certainty = get_survey_sentence_id(sentence_df, context, category)

        for s_age in s_ages:
            small_df = small_childes.loc[small_childes.year.astype(int) == s_age].loc \
                [small_childes.identity == identity].loc[small_childes.category == category].loc[
                small_childes.sentiment == sentiment]
            small_df = small_df.loc[small_df.context.isin([context])]
            small_df['cluster'] = [s_cluster] * len(small_df)
            small_df['weighted foundation'] = [s_f_vote] * len(small_df)
            small_df['central peripheral cluster moral relevance'] = [s_moral_vote] * len(small_df)
            small_df['from survey'] = [True] * len(small_df)
            mfd_df = mfd_df.append(small_df)


    final_df = non_mfd_data.append(mfd_df)
    return final_df



sentence_moral_df = pd.read_csv(f'data/Survey Data/N{version}/sentence_moralization{saving}.csv')

moral_df = pd.read_csv(f'data/Survey Data/N{version}/cluster_moralization{saving}{certain}.csv')
small_childes = read_childes()

new_childes = create_moral_df(small_childes, sentence_moral_df, moral_df)
pickle.dump(new_childes,open(f'./data/v2_moral_df_context_all_moralization_version_N{version}_all.p', 'wb'))



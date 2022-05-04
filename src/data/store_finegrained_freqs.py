from src.data.clustering_models import GMM
from src.data.clean_survey import threshold
from src.data.cluster_childes import map_categories,categories
import pandas as pd
from itertools import product
import os
import pickle

model_name = 'GMM'

def select_childes(data: pd.DataFrame, identity: str, age: int, category: str, sentiment='', use_sentiment=False,
                   exclude_hall: bool = True, reduce = False):

    print(category, sentiment, use_sentiment)
    if exclude_hall:
        data = data.loc[data.corpus != 'Hall']

    categories = []
    #df type

    for foundation, mapped_foundation in map_categories.items():

        if mapped_foundation == category:
            categories.append(foundation)


    if use_sentiment:
        new_data = data.loc[data.identity == identity].loc[data.year.astype(int) == age].loc[
                            data.category.isin(categories)].loc[data.sentiment == sentiment]


    else:
        new_data = data.loc[data.identity == identity].loc[data.year.astype(int) == age].loc[
                            data.category.isin(categories)]

    if reduce:
        # new_data['reduced_sentence'] = new_data.apply(reduce_context(), axis = 1)
        context = list(new_data.reduced_context)
    else:
        context = list(new_data.context)


    return context, new_data


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

def get_moral_count(childes_data:pd.DataFrame, vote,exclude_hall = True, foundation_vote= False):
    if exclude_hall:
        childes_data = childes_data.loc[childes_data.corpus != 'Hall']


    if vote == 'central vote':
        column = 'cluster moral relevance'
        if foundation_vote:
            cat_column = 'central foundation'
        else:
            cat_column = 'category'
    else:
        column = 'central peripheral cluster moral relevance'
        if foundation_vote:
            cat_column = 'weighted foundation'
        else:
            cat_column = 'category'


    non_mfd_data = childes_data.loc[childes_data.year.astype(int) < 7].loc[childes_data.category.isna()]
    moral_relevant_data = childes_data.loc[(childes_data[column] == 'Moral') & (~pd.isna(childes_data.category))]
    moral_irrelevant_data = childes_data.loc[(childes_data[column] == 'Non-moral') & (~pd.isna(childes_data.category))]
    print(len(moral_relevant_data), len(moral_irrelevant_data))
    list_rows = []
    ages = range(1, 7)
    identities = ['child', 'parent']
    for age in ages:
        for identity in identities:
            for c in categories:
                for p in ['pos', 'neg']:

                    all_sentences = childes_data.loc[(childes_data.identity == identity) & (childes_data.year.astype(int) == age)]
                    print(c, p)

                    m_c_df = moral_relevant_data.loc[moral_relevant_data.identity == identity].\
                    loc[moral_relevant_data[cat_column] == c].loc[moral_relevant_data.year.astype(int) == age].loc[moral_relevant_data.sentiment == p]


                    m_ir_df =  moral_irrelevant_data.loc[moral_irrelevant_data.identity ==identity].loc[moral_irrelevant_data.year.astype(int) == age]
                    m_ir_c_df = moral_irrelevant_data.loc[moral_irrelevant_data.identity == identity].loc[
                        moral_irrelevant_data.year.astype(int) == age].loc[moral_irrelevant_data[cat_column] == c].loc[moral_irrelevant_data.sentiment == p]

                    n_mf_df = non_mfd_data.loc[non_mfd_data.identity == identity].loc[non_mfd_data.year.astype(int) == age]


                    row = {'category': c, 'sentiment': p,'moral foundation' : c + ' '+ p, 'age': age, 'identity': identity,
                           'count': len(m_c_df), 'previous count': len(m_c_df) + len(m_ir_c_df),
                           'non mfd': len(n_mf_df), 'non-moral': len(m_ir_df),
                           'sum': len(m_c_df)+ len(n_mf_df)+len(m_ir_df),
                           }

                    list_rows.append(row)

    return pd.DataFrame(list_rows)

def get_cluster_utterances_for_ses_df(data:pd.DataFrame, moral_df, sentence_df, vote, use_foundation_vote = False, certain = False, exclude_hall = True):

    childes_data = data.loc[data.corpus == 'Hall']

    new_df = pd.DataFrame()
    sentences = []

    clusters = []
    cluster_moral_relevance = []
    central_peripheral_cluster_moral_relevance = []
    df_foundations= []
    df_identities = []
    df_sentiments = []
    df_ages = []
    fvote_central = []
    f_vote_weighted = []

    ages = range(1, 7)
    identities = ['child', 'parent']
    sentiments = ['pos', 'neg']
    categories = ['harm', 'authority','fairness', 'loyalty','purity']

    non_mfd_data = childes_data.loc[childes_data.year.astype(int) < 7].loc[childes_data.category.isna()]
    moral_counts = {}
    non_moral_counts = {}
    list_rows = []
    for age, identity, category, sentiment in list(product(ages,identities, categories, sentiments)):
        df = childes_data.loc[(childes_data.identity == identity) & (childes_data.year.astype(int) == age) &
                              (childes_data.category == category) & (childes_data.sentiment == sentiment)]



        for i, row in df.iterrows():
            if (age, identity, category, sentiment) not in moral_counts:
                moral_counts[(age, identity, category, sentiment)] = 0

            if (age, identity, category, sentiment) not in non_moral_counts:
                non_moral_counts[(age, identity, category, sentiment)] = 0

            c = row['cluster']
            if pd.isna(c):
                non_moral_counts[(age, identity, category, sentiment)] += 1
                sentences.append(row['context'])
                clusters.append(c)
                df_ages.append(age)
                df_identities.append(identity)
                cluster_moral_relevance.append('Non-moral')
                central_peripheral_cluster_moral_relevance.append('Non-moral')
                df_foundations.append(category)
                df_sentiments.append(sentiment)
                fvote_central.append('')
                f_vote_weighted.append('')
                continue


            c = int(c)
            central_moralization = get_cluster_moralization(moral_df, category,sentiment, identity, c,
                                                            vote + ' vote')
            if use_foundation_vote:
                foundation = get_cluster_foundation(moral_df, category, sentiment, identity, c,
                                                    vote_column=vote + ' foundation vote')
            else:
                foundation = category

            if (age, identity, foundation, sentiment) not in moral_counts:
                moral_counts[(age, identity, foundation, sentiment)] = 0

            if (age, identity, foundation, sentiment) not in non_moral_counts:
                non_moral_counts[(age, identity, foundation, sentiment)] = 0


            if central_moralization == 'Moral':
                moral_counts[(age, identity, foundation, sentiment)] += 1
            else:
                non_moral_counts[(age, identity, foundation, sentiment)] += 1

            sentences.append(row['context'])
            clusters.append(c)
            cluster_moral_relevance.append(get_cluster_moralization(moral_df, category,sentiment, identity, c,
                                                            'central vote'))
            central_peripheral_cluster_moral_relevance.append(get_cluster_moralization(moral_df, category,sentiment, identity, c,
                                                            'weighted vote'))
            df_foundations.append(category)
            df_identities.append(identity)
            df_ages.append(age)
            df_sentiments.append(sentiment)
            fvote_central.append(get_cluster_foundation(moral_df, category, sentiment, identity, c,
                                                    vote_column= 'central foundation vote'))
            f_vote_weighted.append(get_cluster_foundation(moral_df, category, sentiment, identity, c,
                                                    vote_column= 'weighted foundation vote'))


    for age, identity, category, sentiment in list(product(ages, identities, categories, sentiments)):

        moral_count = 0
        non_moral_count = 0
        if (age, identity, category, sentiment) in moral_counts:
            moral_count = moral_counts[(age, identity, category, sentiment)]
        if (age, identity, category, sentiment) in non_moral_counts:
            non_moral_count = non_moral_counts[(age, identity, category, sentiment)]
        non_mf_df = non_mfd_data.loc[non_mfd_data.identity == identity].loc[non_mfd_data.year.astype(int) == age]

        row = {'category': category, 'sentiment': sentiment, 'moral foundation': category + ' ' + sentiment, 'age': age,
                'identity': identity, 'count': moral_count, 'non mfd': len(non_mf_df), 'non-moral': non_moral_count,
               'sum': len(non_mf_df) + moral_count + non_moral_count}
        list_rows.append(row)

    new_df['context'] = sentences
    new_df['category'] = df_foundations
    new_df['sentiment'] = df_sentiments
    new_df['identity'] = df_identities
    new_df['year'] = df_ages
    new_df['cluster'] = clusters
    new_df['cluster moral relevance'] = cluster_moral_relevance
    new_df['central peripheral cluster moral relevance'] = central_peripheral_cluster_moral_relevance
    new_df['central foundation'] = fvote_central
    new_df['weighted foundation'] = f_vote_weighted

    non_mfd_new_df = non_mfd_data[['context', 'category', 'sentiment', 'identity',
                                'year', 'cluster', 'cluster moral relevance', 'central peripheral cluster moral relevance',
                                'central foundation', 'weighted foundation']]

    new_df = new_df.append(non_mfd_new_df)


    return pd.DataFrame(list_rows), new_df



def count_frequency_with_ses(childes_data: pd.DataFrame,moral_df,sentence_df, vote,exclude_hall=True, foundation_vote = False, certain = False):
    childes_data = childes_data.loc[childes_data.corpus == 'Hall']
    new_childes_data = pd.DataFrame()

    childes_data = childes_data.loc[(childes_data['class'] != '') & (childes_data['race'] != '')]

    races = ['Black', 'White']
    social_class = ['UC', 'WC']
    frequency_df = pd.DataFrame()

    for r in races:
        for s in social_class:
            r_s_childes_data = childes_data.loc[(childes_data.race == r) & (childes_data['class'] == s)]
            print(len(r_s_childes_data))
            moral_count, new_df = get_cluster_utterances_for_ses_df(r_s_childes_data,moral_df,sentence_df, vote, foundation_vote,certain)
            r_s_frequency = create_moral_frequency(moral_count)

            r_s_frequency['race'] = [r] * len(r_s_frequency)
            r_s_frequency['social class'] = [s] * len(r_s_frequency)
            new_df['race'] = [r] * len(new_df)
            new_df['class'] = [s] * len(new_df)
            new_childes_data = new_childes_data.append(new_df)

            frequency_df = frequency_df.append(r_s_frequency)

    return frequency_df, new_childes_data




def count_frequencies_with_gender(childes_data: pd.DataFrame, vote,exclude_hall=True, foundation_vote = False):
    if exclude_hall:
        childes_data = childes_data.loc[childes_data.corpus != 'Hall']


    childes_data = childes_data.loc[childes_data['child gender'] != '']

    frequency_df = pd.DataFrame()
    for gender in ['female', 'male']:
        childes_data_g = childes_data.loc[childes_data['child gender'] == gender]
        gender_frequency_df = create_moral_frequency(get_moral_count( childes_data_g,vote, True, foundation_vote))
        gender_frequency_df['gender'] = [gender] * len(gender_frequency_df)
        frequency_df = frequency_df.append(gender_frequency_df)


    return frequency_df
def create_moral_frequency(frequency_df):
    frequency_df['freq'] = frequency_df['count'] / frequency_df['sum']

    moral_count = {}
    for age in frequency_df.age.unique():
        for identity in ['child', 'parent']:
            moral_count[(age, identity)] = \
            frequency_df.loc[(frequency_df.age == age) & (frequency_df.identity == identity)]['freq'].sum()

    moral_count_col = []

    for i, row in frequency_df.iterrows():
        moral_count_col.append(row['freq'] / moral_count[(row['age'], row['identity'])])

    frequency_df['moral frequency'] = moral_count_col
    return frequency_df


if __name__ == '__main__':

    version = 300
    at1 = False
    c = True
    use_foundation_vote = True
    childes_data_full = pickle.load(open(f'./data/v2_moral_df_context_all_moralization_version_N{version}_all.p', 'rb'))
    saved_cluster = False

    saving = '' if not at1 else '_at1'
    certain = '' if not c else '_certain'
    sentence_moral_df = pd.read_csv(f'data/Survey Data/N{version}/sentence_moralization{saving}.csv')

    moral_df = pd.read_csv(f'data/Survey Data/N{version}/cluster_moralization{saving}{certain}.csv')


    childes_data_full['category'] = [map_categories[c] if c in map_categories.keys() else c for c in childes_data_full['category']]

    votes = ['weighted']
    for vote in votes:


        polarity_frequency = create_moral_frequency(get_moral_count(childes_data_full, True, True, use_foundation_vote))
        polarity_frequency.to_csv(
            f'data/Survey Data/N{version}/freqs/v2_new_frequency_polarity_count_{vote}_full_CHILDES_version_N{version}{saving}{certain}_fvote_{use_foundation_vote}_all.csv')

        gender_frequency = count_frequencies_with_gender(childes_data_full, vote, True, True)
        gender_frequency.to_csv(f'data/Survey Data/N{version}/freqs/v3_new_frequency_gender_count_{vote}_full_CHILDES_version_N{version}{saving}{certain}_fvote_{use_foundation_vote}_all.csv')


        ses_frequency , new_hall_data = count_frequency_with_ses(childes_data_full, moral_df,sentence_moral_df, vote, True, use_foundation_vote, c)
        ses_frequency.to_csv(f'data/Survey Data/N{version}/freqs/v2_new_frequency_ses_count_{vote}_full_CHILDES_version_N{version}{saving}{certain}_fvote_{use_foundation_vote}.csv')
        new_hall_data.to_csv(f'data/hall_moral_df_version_N{saving}{certain}_fvote_{use_foundation_vote}.csv')
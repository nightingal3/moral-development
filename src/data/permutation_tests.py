import pandas as pd
import numpy as np
from src.data.store_finegrained_freqs import create_moral_frequency

categories = ['harm', 'fairness', 'authority', 'purity', 'loyalty']



def count_frequencies_with_polarity(childes_data:pd.DataFrame,childes_word_df, vote,exclude_hall = True, foundation_vote= False):
    if exclude_hall:
        childes_data = childes_data.loc[childes_data.corpus != 'Hall']
        childes_word_df = childes_word_df.loc[childes_word_df.corpus != 'Hall']

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

    print(len(childes_word_df), len(childes_data))
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
                    all_words = childes_word_df.loc[(childes_word_df.identity == identity) & (childes_word_df.year.astype(int) == age)]

                    all_sentences = childes_data.loc[(childes_data.identity == identity) & (childes_data.year.astype(int) == age)]
                    print(c, p)

                    m_c_df = moral_relevant_data.loc[moral_relevant_data.identity == identity].\
                    loc[moral_relevant_data[cat_column] == c].loc[moral_relevant_data.year.astype(int) == age].loc[moral_relevant_data.sentiment == p]


                    m_ir_df =  moral_irrelevant_data.loc[moral_irrelevant_data.identity ==identity].loc[moral_irrelevant_data.year.astype(int) == age]
                    m_ir_c_df = moral_irrelevant_data.loc[moral_irrelevant_data.identity == identity].loc[
                        moral_irrelevant_data.year.astype(int) == age].loc[moral_irrelevant_data[cat_column] == c].loc[moral_irrelevant_data.sentiment == p]

                    n_mf_df = non_mfd_data.loc[non_mfd_data.identity == identity].loc[non_mfd_data.year.astype(int) == age]
                    print(len(m_c_df))

                    row = {'category': c, 'sentiment': p,'moral foundation' : c + ' '+ p, 'age': age, 'identity': identity,
                           'count': len(m_c_df), 'previous count': len(m_c_df) + len(m_ir_c_df),
                           'non mfd': len(n_mf_df), 'non-moral': len(m_ir_df),
                           'sum': len(all_sentences), 'word count':len(all_words)}
                    list_rows.append(row)

    return pd.DataFrame(list_rows)



def count_frequency_with_ses(childes_data: pd.DataFrame, childes_word_df,vote, exclude_hall=False):
    childes_data = childes_data.loc[childes_data.corpus == 'Hall']
    childes_word_df = childes_word_df.loc[childes_word_df.corpus == 'Hall']


    childes_data = childes_data.loc[(childes_data['class'] != '') & (childes_data['race'] != '')]
    childes_word_df = childes_word_df.loc[(childes_word_df['class'] != '') & (childes_word_df['race'] != '')]

    races = ['Black', 'White']
    social_class = ['UC', 'WC']

    frequency_df = pd.DataFrame()

    for r in races:
        for s in social_class:
            r_s_childes_data = childes_data.loc[(childes_data.race == r) & (childes_data['class'] == s)]
            r_s_childes_words_df = childes_word_df.loc[(childes_word_df.race == r) & (childes_word_df['class'] == s)]

            r_s_frequency = create_moral_frequency(
                count_frequencies_with_polarity(r_s_childes_data, r_s_childes_words_df,vote, False))

            r_s_frequency['race'] = [r] * len(r_s_frequency)
            r_s_frequency['social class'] = [s] * len(r_s_frequency)
            frequency_df = frequency_df.append(r_s_frequency)


    return frequency_df




def get_moral_language_frequency(df1, df2, vote, foundation_vote):
    if 'central' in vote:
        column = 'cluster moral relevance'
    else:
        column = 'central peripheral cluster moral relevance'


    df1_moral = len(df1.loc[(df1[column] == 'Moral')].loc[~pd.isna(df1.category)])

    df2_moral = len(df2.loc[(df2[column] == 'Moral') & (~pd.isna(df2.category))])

    df1_moral_frequency = (df1_moral) / len(df1)
    df2_moral_frequency = (df2_moral) / len(df2)

    return abs(df1_moral_frequency - df2_moral_frequency)

def get_category_moral_language_frequency(category):
    def function(df1, df2,vote, foundation_vote):
        if 'central' in vote:
            column = 'cluster moral relevance'
        else:
            column = 'central peripheral cluster moral relevance'

        if foundation_vote:
            if 'central' in vote:
                cat_column = 'central foundation'
            else:
                cat_column = 'weighted foundation'
        else:
            cat_column = 'category'

        df1_moral = (df1.loc[(df1[column] == 'Moral') & (~pd.isna(df1[cat_column]))])
        df1_moral =  len(df1_moral.loc[df1_moral[cat_column] == category])

        df2_moral = (df2.loc[(df2[column] == 'Moral') & (~pd.isna(df2[cat_column]))])
        df2_moral = len(df2_moral.loc[df2_moral[cat_column] == category])

        df1_moral_frequency = (df1_moral) / len(df1)
        df2_moral_frequency = (df2_moral) / len(df2)

        return abs(df1_moral_frequency - df2_moral_frequency)

    return function

#FIX BUG
def get_category_sentiment_moral_language_frequency(category, sentiment):
    def function(df1, df2, vote,foundation_vote):
        if 'central' in vote:
            column = 'cluster moral relevance'
        else:
            column = 'central peripheral cluster moral relevance'

        if foundation_vote:
            if 'central' in vote:
                cat_column = 'central foundation'
            else:
                cat_column = 'weighted foundation'
        else:
            cat_column = 'category'


        df1_moral = (df1.loc[(df1[column] == 'Moral') & (~pd.isna(df1[cat_column]))])
        df1_moral = len(df1_moral.loc[(df1_moral[cat_column] == category) &(df1_moral.sentiment == sentiment)])

        df2_moral = (df2.loc[(df2[column] == 'Moral') & (~pd.isna(df2[cat_column]))])
        df2_moral = len(df2_moral.loc[(df2_moral[cat_column] == category)& (df2_moral.sentiment == sentiment)])

        df1_moral_frequency = (df1_moral) / len(df1)
        df2_moral_frequency = (df2_moral) / len(df2)

        return abs(df1_moral_frequency - df2_moral_frequency)

    return function


def get_category_frequency_sentiment_difference(category):
    def function(df1, df2, vote,foundation_vote):
        if 'central' in vote:
            column = 'cluster moral relevance'
        else:
            column = 'central peripheral cluster moral relevance'

        if foundation_vote:
            if 'central' in vote:
                cat_column = 'central foundation'
            else:
                cat_column = 'weighted foundation'
        else:
            cat_column = 'category'

        df1_moral = (df1.loc[(df1[column] == 'Moral') & (~pd.isna(df1[cat_column]))])
        df1_moral_pos = len(df1_moral.loc[(df1_moral[cat_column] == category) & (df1_moral.sentiment == 'pos')])
        df1_moral_neg = len(df1_moral.loc[(df1_moral[cat_column] == category) & (df1_moral.sentiment == 'neg')])
        if (df1_moral_pos) +(df1_moral_neg) == 0:
            df1_sentiment_difference = float("inf")
        else:
            df1_sentiment_difference = (df1_moral_pos / (df1_moral_pos + df1_moral_neg)) - df1_moral_neg / (df1_moral_pos + df1_moral_neg)

        df2_moral = (df2.loc[(df2[column] == 'Moral') & (~pd.isna(df2[cat_column]))])
        df2_moral_pos = len(df2_moral.loc[(df2_moral[cat_column] == category) & (df2_moral.sentiment == 'pos')])
        df2_moral_neg = len(df2_moral.loc[(df2_moral[cat_column] == category) & (df2_moral.sentiment == 'neg')])


        if (df2_moral_neg) + (df2_moral_pos) == 0:
            df2_sentiment_difference = float("inf")
        else:
            df2_sentiment_difference = (df2_moral_pos / (df2_moral_pos + df2_moral_neg)) - df2_moral_neg / (
                    df2_moral_pos + df2_moral_neg)
        print(abs(df1_sentiment_difference - df2_sentiment_difference))
        return abs(df1_sentiment_difference - df2_sentiment_difference)

    return function

def get_moral_frequency_sentiment_difference(df1, df2,vote, foundation_vote):

    if 'central' in vote:
        column = 'cluster moral relevance'
    else:
        column = 'central peripheral cluster moral relevance'
    df1_moral = (df1.loc[(df1[column] == 'Moral') & (~pd.isna(df1.category))])
    df1_moral_pos = len((df1_moral.loc[df1_moral.sentiment == 'pos']))
    df1_moral_neg = len((df1_moral.loc[df1_moral.sentiment == 'neg']))
    if (df1_moral_pos) + df1_moral_neg == 0:

        df1_sentiment_difference = float("inf")
    else:
        df1_sentiment_difference = (df1_moral_pos / (df1_moral_pos + df1_moral_neg)) - df1_moral_neg / (df1_moral_pos + df1_moral_neg)

    df2_moral = (df2.loc[(df2[column] == 'Moral') & (~pd.isna(df2.category))])
    df2_moral_pos = len((df2_moral.loc[df2_moral.sentiment == 'pos']))
    df2_moral_neg = len((df2_moral.loc[df2_moral.sentiment == 'neg']))

    if (df2_moral_pos) + df2_moral_neg == 0:
        df2_sentiment_difference = float("inf")
    else:
        df2_sentiment_difference = (df2_moral_pos / (df2_moral_pos + df2_moral_neg)) - df2_moral_neg / (
                df2_moral_pos + df2_moral_neg)

    return abs(df1_sentiment_difference - df2_sentiment_difference)



def get_sentiment_moral_language_frequency(category, sentiment):
    def function(df1, df2, vote,foundation_vote):
        if 'central' in vote:
            column = 'cluster moral relevance'
        else:
            column = 'central peripheral cluster moral relevance'
        if foundation_vote:
            if 'central' in vote:
                cat_column = 'central foundation'
            else:
                cat_column = 'weighted foundation'
        else:
            cat_column = 'category'

        df1_moral = (df1.loc[df1.loc[(df1[column] == 'Moral') & (~pd.isna(df1[cat_column]))]])
        df1_moral = len(df1_moral.loc[(df1_moral[cat_column] == category) &(df1_moral.sentiment == sentiment)])

        df2_moral = (df2.loc[df2.loc[(df2[column] == 'Moral') & (~pd.isna(df2[cat_column]))]])
        df2_moral = len(df2_moral.loc[(df2_moral[cat_column] == category)& (df2_moral.sentiment == sentiment)])

        df1_moral_frequency = (df1_moral) / len(df1)
        df2_moral_frequency = (df2_moral) / len(df2)
        return abs(df1_moral_frequency - df2_moral_frequency)

    return function






def permutation_test(df1, df2, vote,foundation_vote,diff,function, permutation_num = 1000):
    n1 , n2 = len(df1), len(df2)
    total_df = df1.copy(deep = True).append(df2.copy(deep = True))
    zs = 0
    for n in range(permutation_num):
        df = total_df.sample(frac=1).reset_index(drop=True)

        new_df1 = df.iloc[:n1, :]
        new_df2 = df.iloc[n1:, :]
        zs += (diff < function(new_df1, new_df2, vote, foundation_vote))


    return zs / permutation_num






def ses_statistical_difference(childes_data, vote, foundation_vote):


    list_rows = []

    races = ['Black', 'White']
    social_class = ['UC', 'WC']

    for identity in ['child', 'parent']:
        for r in races:
            print(r)

            r_UC_childes_data = childes_data.loc[(childes_data.race == r) & (childes_data['class'] == 'UC') & (childes_data.identity == identity)]
            r_WC_childes_data = childes_data.loc[(childes_data.race == r) & (childes_data['class'] == 'WC') & (childes_data.identity == identity)]

            moral_frequency_difference = get_moral_language_frequency(r_UC_childes_data, r_WC_childes_data, vote, foundation_vote)

            moral_permutation_value = permutation_test(r_UC_childes_data,r_WC_childes_data,
                                                       vote,foundation_vote, moral_frequency_difference,get_moral_language_frequency)
            print(moral_permutation_value,moral_frequency_difference)
            row = {'identity': identity, 'constant variable': 'race', 'constant value': r, 'category' : 'all', 'sentiment': 'all',
                   'test' :'moral language difference', 'value': moral_frequency_difference, 'significance': moral_permutation_value}
            list_rows.append(row)



            categories_ps = {}
            category_sentiment_pcs = {}
            category_sentiment_difference = {}
            for c in categories:



                sentiment_difference = get_category_frequency_sentiment_difference(c)(r_UC_childes_data, r_WC_childes_data, vote, foundation_vote)
                permutation_value = permutation_test(r_UC_childes_data, r_WC_childes_data, vote,foundation_vote,
                                                     sentiment_difference,
                                                     get_category_frequency_sentiment_difference(c))

                category_sentiment_difference[c] = permutation_value
                row = {'identity': identity, 'constant variable': 'race', 'constant value': r, 'category': c,
                       'sentiment': 'all',
                       'test': 'foundation sentiment difference', 'value': sentiment_difference,
                       'significance': category_sentiment_difference[c]}
                list_rows.append(row)
                print(c,category_sentiment_difference[c],sentiment_difference )


                for p in ['pos', 'neg']:
                    moral_frequency_difference = get_category_sentiment_moral_language_frequency(c, p)(r_UC_childes_data,
                                                                                                       r_WC_childes_data, vote, foundation_vote)
                    permutation_value = permutation_test(r_UC_childes_data, r_WC_childes_data, vote,foundation_vote,
                                                         moral_frequency_difference,
                                                         get_category_sentiment_moral_language_frequency(c, p))
                    category_sentiment_pcs[c,p] = permutation_value

                    row = {'identity': identity, 'constant variable': 'race', 'constant value': r, 'category': c,
                           'sentiment': p,
                           'test': 'full foundation difference', 'value': moral_frequency_difference,
                           'significance': category_sentiment_pcs[c, p]}
                    list_rows.append(row)
                    print(c, p,category_sentiment_pcs[c, p] ,moral_frequency_difference)





        for sc in social_class:
            print(sc)
            sc_black_childes_data =childes_data.loc[(childes_data.race == 'Black') & (childes_data['class'] == sc) & (childes_data.identity == identity)]
            sc_white_childes_data = childes_data.loc[(childes_data.race == 'White') & (childes_data['class'] == sc) & (childes_data.identity == identity)]

            moral_frequency_difference = get_moral_language_frequency(sc_black_childes_data, sc_white_childes_data, vote, foundation_vote)
            moral_permutation_value = permutation_test(sc_black_childes_data, sc_white_childes_data, vote,foundation_vote,
                                                       moral_frequency_difference, get_moral_language_frequency)

            row = {'identity': identity, 'constant variable': 'class', 'constant value': sc, 'category': 'all',
                   'sentiment': 'all',
                   'test': 'moral language difference', 'value': moral_frequency_difference,
                   'significance': moral_permutation_value}
            list_rows.append(row)


            moral_sentiment_difference = get_moral_frequency_sentiment_difference(sc_black_childes_data, sc_white_childes_data,
                                                                                  vote, foundation_vote)
            sentiment_difference_permutation_value = permutation_test(sc_black_childes_data, sc_white_childes_data, vote,foundation_vote,
                                                                      moral_sentiment_difference,
                                                                      get_moral_frequency_sentiment_difference)

            row = {'identity': identity, 'constant variable': 'class', 'constant value': sc, 'category': 'all',
                   'sentiment': 'all',
                   'test': 'sentiment difference', 'value': moral_sentiment_difference,
                   'significance': sentiment_difference_permutation_value}
            list_rows.append(row)

            categories_ps = {}
            category_sentiment_pcs = {}
            category_sentiment_difference = {}
            for c in categories:

                sentiment_difference = get_category_frequency_sentiment_difference(c)(sc_black_childes_data, sc_white_childes_data, vote, foundation_vote)
                permutation_value = permutation_test(sc_black_childes_data, sc_white_childes_data, vote,foundation_vote,
                                                     sentiment_difference,
                                                     get_category_frequency_sentiment_difference(c))

                category_sentiment_difference[c] = permutation_value

                row = {'identity': identity, 'constant variable': 'class', 'constant value': sc, 'category': c,
                       'sentiment': 'all',
                       'test': 'foundation sentiment difference', 'value': sentiment_difference,
                       'significance': category_sentiment_difference[c]}
                list_rows.append(row)


                for p in ['pos', 'neg']:


                    moral_frequency_difference = get_category_sentiment_moral_language_frequency(c, p)(sc_black_childes_data, sc_white_childes_data,
                                                                                                       vote, foundation_vote)
                    permutation_value = permutation_test(sc_black_childes_data, sc_white_childes_data, vote,foundation_vote,
                                                         moral_frequency_difference,
                                                         get_category_sentiment_moral_language_frequency(c, p))
                    category_sentiment_pcs[c, p] = permutation_value

                    row = {'identity': identity, 'constant variable': 'class', 'constant value': sc, 'category': c,
                           'sentiment': p,
                           'test': 'full foundation difference', 'value': moral_frequency_difference,
                           'significance': category_sentiment_pcs[c, p]}
                    list_rows.append(row)
    df = pd.DataFrame(list_rows)
    return df


if __name__ == '__main__':
    version = 300


    saving = ''
    certain = '_certain'
    use_foundation_vote = True
    childes_data_full = pd.read_csv(f'data/hall_moral_df_version_N{saving}{certain}_fvote_{use_foundation_vote}.csv')
    categories = ['harm', 'fairness', 'purity']

    #testing race and social class
    #central vote
    vote = 'weighted'
    significance_df = ses_statistical_difference(childes_data_full, vote, use_foundation_vote)
    significance_df.to_csv(f'./data/Survey Data/N{version}/small_ses_significance_{vote}_full_CHILDES_version_N{version}{saving}{certain}{use_foundation_vote}.csv',
                           index=False)


import pandas as pd
import numpy as np
import os
import pickle
from collections import Counter
from functools import reduce
import operator
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff


from nltk.tokenize import sent_tokenize

__dir__ = 'data/'
threshold = 0.7
c_p_ratio = 1
necessary_vote_num = 1
kappa_need = 3
foundation_map = {'Care/harm': 'harm', 'Purity/degradation': 'purity', 'Authority/subversion': 'authority', 'Loyalty/betrayal': 'loyalty', 'Fairness/cheating': 'fairness',
                  'None of these': 'none'}


def get_attentive_survey(survey_df : pd.DataFrame, at1 = False):

    new_df = survey_df.copy(deep=True)

    attention_columns = ['AC1 response_1',	'AC1 response_2',	'AC1 response_3',	'AC2']


    ac1_df = new_df.loc[(new_df['AC1 response_1'] == 'Disagree,Neither agree nor disagree,Agree') &
                        (new_df['AC1 response_2'] == 'Disagree,Neither agree nor disagree,Agree') &
                          (new_df['AC1 response_3'] == 'Disagree,Neither agree nor disagree,Agree')]
    at1_fail = len(new_df) - len(ac1_df)



    if not at1:
        ac1_df = new_df

    ac1_df = ac1_df.reset_index(drop=True)
    ac1_df['answer_num'] =  [len(sent_tokenize(s)) if not pd.isna(s) else 0 for s in list(ac1_df['AC2'])]

    ac2_df = ac1_df.loc[ac1_df['answer_num'] >= 2]
    ac2_df= ac2_df.reset_index(drop = True)

    at2_fail = len(ac1_df) - len(ac2_df)
    return ac2_df, at1_fail, at2_fail

def get_uncertain_df(df, threshold):
        moral_df = df.loc[df['moral vote'] == 'Moral']
        uncertain_df = moral_df.loc[moral_df['moral vote number'] / moral_df['total vote number'] < threshold]
        certain_df = moral_df.loc[moral_df['moral vote number'] / moral_df['total vote number'] >= threshold]
        return uncertain_df, certain_df

def get_unmatching_df(df):
    matching_df = df.loc[df['foundation'] == df['foundation vote']]
    unmatching_df = df.loc[df['foundation'] != df['foundation vote']]
    return unmatching_df, matching_df

def get_certain_survey(sentence_moralization_df:pd.DataFrame, threshold = threshold):
    non_moral_sentences = sentence_moralization_df.loc[sentence_moralization_df['moral vote'] == 'Non-moral']
    uncertain_df, certain_df = get_uncertain_df(sentence_moralization_df, threshold)

    #for uncertain_df we only keep those that match MFD

    unmatching_df , matching_df = get_unmatching_df(uncertain_df)
    new_df = certain_df.append(matching_df).append(non_moral_sentences)

    return list(new_df['id'])



def clean_survey(survey_df : pd.DataFrame, question_num = 1371, at1 = False):
    '''

    - Filters out responses from RAs
    - Only English speakers
    - Removes those failing the attention task
    '''
    new_df = survey_df.copy(deep=True)

    remove_ra_df = new_df.loc[(new_df.Finished.isin(['True', True, 'TRUE'])) & (~new_df['Prolific ID'].isna()) & (new_df['Prolific ID'] != 'testing')]

    remove_ra_df = remove_ra_df.reset_index(drop = True)
    print('Remove ra:', len(new_df) - len(remove_ra_df))

    english_df1 = remove_ra_df.loc[(remove_ra_df['english first'] == 'Yes')]
    english_df2 = remove_ra_df.loc[~pd.isna(remove_ra_df['english years'])]
    english_df2 = english_df2.loc[english_df2['english years'].astype(int) >= 10]
    english_df = english_df1.append(english_df2)
    print('remove english:', len(remove_ra_df) - len(english_df))

    english_df = english_df.reset_index(drop=True)

    attentive_df ,at1_fail, at2_fail = get_attentive_survey(english_df, at1)
    return attentive_df ,at1_fail, at2_fail



def duplicate_column(questions_df, column):
    question_column = list(questions_df[column])
    question_column = list(np.repeat(question_column, 2))
    return question_column


def get_answers(survey_df: pd.DataFrame, questions_df: pd.DataFrame, question_num = 1371) -> pd.DataFrame:
    df = survey_df.loc[survey_df.Finished.isin(['True',True])]
    question_columns = list(survey_df.columns)[22: 22 + question_num * 2]
    df = df.loc[:, question_columns]
    df = df.T
    questions_text = list(questions_df.utterance)
    questions_text = [str.replace(str[:str.index(':') + 2], '') for str in questions_text]
    questions_text = list(np.repeat(questions_text, 2))
    df['sentence'] = questions_text
    df['type'] = duplicate_column(questions_df, 'utterance type')
    df['cluster'] = duplicate_column(questions_df, 'label')
    df['foundation'] = duplicate_column(questions_df, 'foundation')
    df['identity'] = duplicate_column(questions_df, 'identity')
    df['id'] = duplicate_column(questions_df, 'sentence_id')
    df = df.reset_index()
    return df


def get_utterance_morality(total_df:pd.DataFrame) -> (pd.DataFrame, int):

    agreement_count = 0
    all = 0
    list_rows = []
    sentences = total_df['id'].unique()


    all_columns = list(total_df.columns)
    answer_columns = [i for i, x in enumerate(all_columns) if
                      x not in ['sentence', 'type', 'cluster', 'foundation', 'identity', 'id', 'index']]

    for id in sentences:

        sentence_df = total_df.loc[total_df.id == id]

        moral_df = sentence_df.iloc[[0], answer_columns]
        cluster = list(sentence_df['cluster'])[0]

        sentence = list(sentence_df['sentence'])[0]

        identity = list(sentence_df['identity'])[0]

        answer_num =int(moral_df.fillna(0).replace('Moral', 1).replace('Non-moral', 1).sum(axis=1))
        moral_num = int(moral_df.fillna(0).replace('Moral', 1).replace('Non-moral', 0).sum(axis=1))

        if len(list(moral_df.mode(axis = 1))) >= necessary_vote_num and answer_num >= necessary_vote_num:
            moral_vote = list(moral_df.mode(axis = 1)[0])[0]

            foundation_df =  sentence_df.iloc[[1], answer_columns]
            foundations = list(foundation_df.iloc[0])
            foundations = [x for x in foundations if (not pd.isna(x)) and x not in ['Not sure']]
            if len(foundations) == 0:

                f_majority = ''

            else:
                foundations = [x.split(',') for x in foundations]

                foundations = reduce(operator.concat, foundations)
                foundations = [foundation_map[x]  for x in foundations if x not in ['Not sure', ' but still moral']]
                f_counter = Counter(foundations)
                f_majority = max(f_counter.items(), key = lambda x : x[1])[0]
        else:
            # moral_vote = 'None'
            moral_vote = 'Non-moral'
            f_majority = ''

        row = {'id': id, 'moral vote': moral_vote, 'foundation vote': f_majority,
               'sentence' :sentence, 'cluster': cluster, 'identity': identity, 'total vote number' : answer_num,'moral vote number': moral_num,
               'foundation': list(sentence_df.foundation)[0].split()[0], 'sentiment': list(sentence_df.foundation)[0].split()[1]}
        list_rows.append(row)

        if moral_vote == 'Moral' and f_majority != '':
            all += 1
            if f_majority == row['foundation']:
                agreement_count += 1





    return pd.DataFrame(list_rows), agreement_count / all


def get_cluster_foundation_vote(central, central_moral_vote, peripheral,peripheral_moral_vote, answer_columns):
    central_idx = [i * 2 + 1 for i in range(int(len(central) / 2))]
    peripheral_idx = [i * 2 + 1 for i in range(int(len(peripheral) / 2))]
    f_counter = Counter([])
    f_majority_central = ''



    foundation_central = central.iloc[central_idx, answer_columns]


    if len(foundation_central) > 0 and len(list(foundation_central.mode(axis = 1))) >= 1:


        foundations = list(foundation_central.mode(axis = 1)[0])
        foundations = [x for x in foundations if (not pd.isna(x)) and x not in ['Not sure']]
        foundations = [x.split(',') for x in foundations]
        if len(foundations) > 0:

            foundations = reduce(operator.concat, foundations)
            foundations = [foundation_map[x] for x in foundations if x not in ['Not sure', ' but still moral']]
            f_counter = Counter(foundations)
            if len(f_counter.items()) > 0:
                f_majority_central = max(f_counter.items(), key=lambda x: x[1])[0]

    f_majority_both = f_majority_central
    foundation_peripheral = peripheral.iloc[peripheral_idx, answer_columns]


    if len(foundation_peripheral) > 0 and len(list(foundation_peripheral.mode(axis = 1))) >= 1:

        foundations = list(foundation_peripheral.mode(axis=1)[0])
        foundations = [x for x in foundations if (not pd.isna(x)) and x not in ['Not sure']]
        foundations = [x.split(',') for x in foundations]

        if len(foundations) > 0:
            foundations = reduce(operator.concat, foundations)
            foundations = [foundation_map[x] for x in foundations if x not in ['Not sure', ' but still moral']]
            f_counter_both = Counter(foundations) + f_counter
            if len(f_counter_both.items()) > 0:
                f_majority_both = max(f_counter_both.items(), key=lambda x: x[1])[0]



    return f_majority_central, f_majority_both


def get_cluster_moralization(total_df:pd.DataFrame, certain_ids = []):
    list_rows = []
    foundations = total_df.foundation.unique()
    all_columns = list(total_df.columns)
    answer_columns = [i for i , x in enumerate(all_columns) if x not in ['sentence', 'type', 'cluster', 'foundation', 'identity', 'id', 'index']]
    for identity in ['child', 'parent']:
        df = total_df.loc[total_df.identity == identity]
        for f in foundations:
            f_questions = df.loc[df.foundation == f]
            clusters = f_questions.cluster.unique()
            for c in clusters:


                f_c_questions = f_questions.loc[f_questions.cluster == c]
                if certain_ids:
                    f_c_questions = f_c_questions.loc[f_c_questions.id.isin(certain_ids)]

                central = f_c_questions.loc[f_c_questions.type == 'central']
                peripheral = f_c_questions.loc[f_c_questions.type == 'peripheral']

                moral_central_idx = [i * 2 for i in range(int(len(central) / 2))]
                moral_peripheral_idx = [i * 2 for i in range(int(len(peripheral) / 2))]


                moral_central = central.iloc[moral_central_idx, answer_columns]
                moral_central['answer_num'] = moral_central.fillna(0).replace('Moral', 1).replace('Non-moral', 1).sum(axis = 1)

                moral_central = moral_central.loc[moral_central['answer_num'] >= 1] #TODO change to 2
                moral_central = moral_central.drop(columns = ['answer_num'])


                if len(list(moral_central.mode(axis = 1))) >= necessary_vote_num and len(moral_central) > 0:

                    moral_central['moral vote'] = moral_central.mode(axis = 1)[0]


                    central_vote = Counter(moral_central['moral vote'])

                    central_vote = {k :v * c_p_ratio for k , v in central_vote.items() if not pd.isna(k)}

                    c_vote = max(central_vote.keys(), key = lambda x: central_vote[x])

                else:

                    central_vote =  Counter([])
                    # c_vote = 'None'
                    c_vote = 'Non-moral'
                    moral_central['moral vote'] = ['Non-moral'] * len(moral_central)

                moral_peripheral = peripheral.iloc[moral_peripheral_idx, answer_columns]
                moral_peripheral['answer_num'] = moral_peripheral.fillna(0).replace('Moral', 1).replace('Non-moral', 1).sum(axis=1)
                moral_peripheral = moral_peripheral.loc[moral_peripheral['answer_num'] >= 1] #TODO change to 2
                moral_peripheral = moral_peripheral.drop(columns=['answer_num'])

                if len(moral_peripheral) > 0 and len(list(moral_peripheral.mode(axis = 1))) >= necessary_vote_num:

                    moral_peripheral['moral vote'] = moral_peripheral.mode(axis = 1)[0]

                    peripheral_vote = Counter(moral_peripheral['moral vote'])
                    peripheral_vote = {k: v for k, v in peripheral_vote.items() if not pd.isna(k)}



                    p_vote = max(peripheral_vote.keys(), key = lambda x: peripheral_vote[x])

                else:
                    moral_peripheral['moral vote'] = ['Non-moral'] * len(moral_peripheral)
                    # p_vote = 'None'
                    p_vote = 'Non-moral'
                    peripheral_vote = Counter([])



                collective_vote = Counter(central_vote) + Counter(peripheral_vote)

                if len(collective_vote) >= 1:

                    final_vote = max((collective_vote).items(), key = lambda x : x[1])[0]

                else:

                    final_vote = None

                f_majority_central, f_majority_both = get_cluster_foundation_vote(central,
                                                                                  moral_central['moral vote'],
                                                                                  peripheral, moral_peripheral['moral vote'], answer_columns)
                row = {'identity': identity, 'foundation': f, 'cluster': c, 'central sentences': len(moral_central_idx),
                       'peripheral sentences': len(moral_peripheral_idx), 'central vote': c_vote,
                       'peripheral vote':  p_vote, 'weighted vote': final_vote,'central foundation vote': f_majority_central, 'weighted foundation vote': f_majority_both}
                list_rows.append(row)

    return pd.DataFrame(list_rows)





if __name__ == '__main__':


    #Cleans the survey data

    version = 300
    survey_df = pd.read_csv(f'data/Survey Data/N{version}/survey.csv')
    questions_df = pd.read_csv('data/cluster_labels_dimension_reduced_large.csv')


    certain = '_certain'

    clean_df, at1_n, at2_n = clean_survey(survey_df, at1 = False)
    print('failing AC1: ', at1_n, 'failing AC2:', at2_n)

    final_df = get_answers(clean_df, questions_df)


    final_df.to_csv(f'data/Survey Data/N{version}/survey_cleaned.csv', index=False)


    # moral clustering results

    final_df = pd.read_csv(f'data/Survey Data/N{version}/survey_cleaned.csv')
    sentence_moral_df, agreement_count = get_utterance_morality(final_df)

    sentence_moral_df.to_csv(f'data/Survey Data/N{version}/sentence_moralization.csv')
    print(f'foundation agreement: {round(agreement_count * 100, 2)}%')  # N 200 51.88%, 45.39


    certain_ids = get_certain_survey(sentence_moral_df)

    moral_df = get_cluster_moralization(final_df, certain_ids)

    moral_df.to_csv(f'data/Survey Data/N{version}/cluster_moralization{certain}.csv', index=False)

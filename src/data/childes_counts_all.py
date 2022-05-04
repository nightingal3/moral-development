from collections import Counter
import csv
from random import shuffle
import os
import string
from typing import List

from nltk.stem import WordNetLemmatizer
import pandas as pd
import pylangacq
import pdb
import pickle

from src.data.mfd_read import read_mfd

corpora_eng = [
    "Bates.zip",
    "Bernstein.zip",
    "Bliss.zip",
    "Bloom.zip",
    "Bohannon.zip",
    "Braunwald.zip",
    "Brent.zip",
    "Brown.zip",
    "Clark.zip",
    "Demetras1.zip",
    "Demetras2.zip",
    "Evans.zip",
    "Feldman.zip",
    "Garvey.zip",
    "Gathercole.zip",
    "Gelman.zip",
    "Gleason.zip",
    "Gopnik.zip",
    "HSLLD.zip",
    "Haggerty.zip",
    "Hall.zip",
    "Hicks.zip",
    "Higginson.zip",
    "Kuczaj.zip",
    "MacWhinney.zip",
    "McCune.zip",
    "McMillan.zip",
    "Morisset.zip",
    "Nelson.zip",
    "NewEngland.zip",
    "NewmanRatner.zip",
    "Peters.zip",
    "PetersonMcCabe.zip",
    "Post.zip",
    "Rollins.zip",
    "Sachs.zip",
    "Sawyer.zip",
    "Snow.zip",
    "Soderstrom.zip",
    "Sprott.zip",
    "Suppes.zip",
    "Tardif.zip",
    "Valian.zip",
    "VanHouten.zip",
    "VanKleeck.zip",
    "Warren.zip",
    "Weist.zip"
]

corpora_clinical =["ENNI.zip"]

columns = ['words', 'pos', 'child_gender','gender', 'group', 'ses','education','custom']





def read_all_corpora(corpora_list: List, out_filename: str = "data/childes-dict.p", append_pos: bool = False,
                     lemmatize: bool = True) -> tuple:
    """Reads a specified list of corpora from CHILDES and returns data tagged by age.

    Args:
        corpora_list (List): List of corpora names (default listed above)
        out_filename (str): Location to write out final dict
        append_pos (bool): whether to store just the words in an utterance or words and POS
    Returns:
        tuple[dict]: {corpus name: {age: [...word/(word, POS, gender of child/parent)...]}} for children and parents
    """
    corpora_child = {}
    corpora_parents = {}
    if lemmatize:
        lemmatizer = WordNetLemmatizer()

    for corpus in corpora_list:
        print(corpus)
        url = f"{base_url}/{corpus}"
        chats = pylangacq.read_chat(url)
        ages = chats.ages()
        headers = chats.headers()
        if append_pos:
            tokens_by_files_chi = chats.tokens(participants="CHI", by_files=True)
            tokens_by_files_mother = chats.tokens(participants="MOT", by_files=True)
            tokens_by_files_father = chats.tokens(participants="FAT", by_files=True)

            corpora_child[corpus[:-4]] = {}
            corpora_parents[corpus[:-4]] = {}

            for age, header, tokens_child, tokens_mother, tokens_father in zip(ages, headers, tokens_by_files_chi,
                                                                               tokens_by_files_mother,
                                                                               tokens_by_files_father):
                if "CHI" in header["Participants"]:
                    child_gender = header["Participants"]["CHI"]["sex"]

                for item_c in tokens_child:
                    child_word, child_pos = item_c.word, item_c.pos
                    if child_word in string.punctuation or child_word == "CLITIC":
                        continue
                    if lemmatize:
                        child_word = lemmatizer.lemmatize(child_word)

                    if age in corpora_child[corpus[:-4]]:
                        corpora_child[corpus[:-4]][age].append((child_word, child_pos, child_gender))
                    else:
                        corpora_child[corpus[:-4]][age] = [(child_word, child_pos, child_gender)]

                for item_p in tokens_mother:
                    parent_word, parent_pos = item_p.word, item_p.pos
                    if parent_word in string.punctuation or child_word == "CLITIC":
                        continue
                    if lemmatize:
                        parent_word = lemmatizer.lemmatize(parent_word)
                    if age in corpora_parents[corpus[:-4]]:
                        corpora_parents[corpus[:-4]][age].append((parent_word, parent_pos, "female"))
                    else:
                        corpora_parents[corpus[:-4]][age] = [(parent_word, parent_pos, "female")]

                for item_p in tokens_father:
                    parent_word, parent_pos = item_p.word, item_p.pos
                    if parent_word in string.punctuation or child_word == "CLITIC":
                        continue
                    if lemmatize:
                        parent_word = lemmatizer.lemmatize(parent_word)
                    if age in corpora_parents[corpus[:-4]]:
                        corpora_parents[corpus[:-4]][age].append((parent_word, parent_pos, "male"))
                    else:
                        corpora_parents[corpus[:-4]][age] = [(parent_word, parent_pos, "male")]

        else:
            words_by_files_chi = chats.words(participants="CHI", by_files=True)
            words_by_files_pa = chats.words(participants={"MOT", "FAT"}, by_files=True)

            corpora_child[corpus[:-4]] = {}
            corpora_parents[corpus[:-4]] = {}
            for age, words_child, words_parents in zip(ages, words_by_files_chi, words_by_files_pa):
                if age in corpora_child[corpus[:-4]]:
                    corpora_child[corpus[:-4]][age].extend(words_child)
                    corpora_parents[corpus[:-4]][age].extend(words_parents)
                else:
                    corpora_child[corpus[:-4]][age] = words_child
                    corpora_parents[corpus[:-4]][age] = words_parents

    with open(out_filename, "wb") as out_f:
        combined = {"parent": corpora_parents, "child": corpora_child}
        pickle.dump(combined, out_f)

    return corpora_child, corpora_parents


def read_all_corpora_by_utterance(corpora_list: List, out_filename: str = "./data/childes-dict-utterances.p",
                                  append_pos: bool = False, lemmatize: bool = False) -> tuple:
    corpora_child = {}
    corpora_parents = {}
    if lemmatize:
        lemmatizer = WordNetLemmatizer()

    for corpus in corpora_list:

        print(corpus)


        url = f"{base_url}/{corpus}"
        chats = pylangacq.read_chat(url)


        ages = chats.ages()
        headers = chats.headers()

        tokens_by_files_chi = chats.tokens(participants="CHI", by_files=True, by_utterances=True)
        tokens_by_files_mother = chats.tokens(participants="MOT", by_files=True, by_utterances=True)
        tokens_by_files_father = chats.tokens(participants="FAT", by_files=True, by_utterances=True)

        corpora_child[corpus[:-4]] = {}
        corpora_parents[corpus[:-4]] = {}


        for age, header, tokens_child, tokens_mother, tokens_father in zip(ages, headers, tokens_by_files_chi,
                                                                           tokens_by_files_mother,
                                                                       tokens_by_files_father):

            race,family_class = '',''



            if "CHI" in header["Participants"]:
                child_gender = header["Participants"]["CHI"]["sex"]
                if header["Participants"]["CHI"]["ses"] != '':
                    print(header["Participants"]["CHI"]["ses"])
                    splitted = header["Participants"]["CHI"]["ses"].split(',')
                    if len(splitted) == 1:
                        family_class = splitted[0]
                    else:
                        race,family_class = splitted

            for utterance in tokens_child:
                utterance_data = []
                for item_c in utterance:
                    child_word, child_pos = item_c.word, item_c.pos
                    if child_word in string.punctuation or child_word == "CLITIC":
                        continue
                    if lemmatize:
                        child_word = lemmatizer.lemmatize(child_word)

                    utterance_data.append((child_word, child_pos, child_gender, child_gender, race,family_class ))

                if utterance_data == []:
                    continue
                if age in corpora_child[corpus[:-4]]:
                    corpora_child[corpus[:-4]][age].append(utterance_data)
                else:
                    corpora_child[corpus[:-4]][age] = [utterance_data]

            for utterance in tokens_mother:
                utterance_data = []
                for item_p in utterance:
                    parent_word, parent_pos = item_p.word, item_p.pos
                    if parent_word in string.punctuation or child_word == "CLITIC":
                        continue
                    if lemmatize:
                        parent_word = lemmatizer.lemmatize(parent_word)

                    utterance_data.append((parent_word, parent_pos, child_gender, 'female',race, family_class ))

                if utterance_data == []:
                    continue

                if age in corpora_parents[corpus[:-4]]:
                    corpora_parents[corpus[:-4]][age].append(utterance_data)
                else:
                    corpora_parents[corpus[:-4]][age] = [utterance_data]

            for utterance in tokens_father:
                utterance_data = []

                for item_p in utterance:
                    parent_word, parent_pos = item_p.word, item_p.pos
                    if parent_word in string.punctuation or child_word == "CLITIC":
                        continue
                    if lemmatize:
                        parent_word = lemmatizer.lemmatize(parent_word)

                    utterance_data.append((parent_word, parent_pos, child_gender, 'male',race, family_class))
                if utterance_data == []:
                    continue

                if age in corpora_parents[corpus[:-4]]:
                    corpora_parents[corpus[:-4]][age].append(utterance_data)
                else:
                    corpora_parents[corpus[:-4]][age] = [utterance_data]

    with open(out_filename, "wb") as out_f:
        combined = {"parent": corpora_parents, "child": corpora_child}
        pickle.dump(combined, out_f)

    return corpora_child, corpora_parents

def moral_categories_over_time(parent_child_dict: dict, pure_categories: dict, wildcard_categories: dict,
                               with_pos: bool = False) -> pd.DataFrame:
    years = []
    corpora = []
    words = []
    pos = []
    gender = []
    child_gender = []
    race = []
    family_class = []
    speech_categories = []
    identity = []
    sentiment = []
    moral_type = []

    for corpus in parent_child_dict["parent"]:
        parent_dict = parent_child_dict["parent"]
        child_dict = parent_child_dict["child"]
        print(corpus)
        print("child")
        for age in child_dict[corpus]:
            print(age)
            year, month, _ = age
            year_frac = year + (month / 12)
            for utterance in child_dict[corpus][age]:




                if with_pos:
                    child_no_punct = [item for item in utterance if item[0] not in string.punctuation]
                else:
                    child_no_punct = [item for item in utterance if item not in string.punctuation]


                for item in child_no_punct:
                    if with_pos:
                        word = item[0]

                    else:
                        word = item
                    category = None
                    if word in pure_categories:
                        category = pure_categories[word]
                    elif any(key[:-1] in word for key in wildcard_categories):
                        for key in wildcard_categories:
                            if key[:-1] in word:
                                category = wildcard_categories[key]
                    if category == None:
                        print(item)
                        years.append(year_frac)
                        child_gender.append(item[2])

                        gender.append(item[3])
                        race.append(item[4])
                        family_class.append(item[5])
                        corpora.append(corpus)
                        if with_pos:
                            words.append(item[0])
                            pos.append(item[1])
                        else:
                            words.append(item)


                        speech_categories.append(None)
                        identity.append("child")
                        sentiment.append(None)
                        moral_type.append(None)
                    else:
                        for cat in category:
                            years.append(year_frac)
                            corpora.append(corpus)
                            if with_pos:
                                words.append(item[0])
                                pos.append(item[1])
                            else:
                                words.append(item)

                            print(item)
                            child_gender.append(item[2])
                            gender.append(item[3])
                            race.append(item[4])
                            family_class.append(item[5])
                            speech_categories.append(cat)
                            identity.append("child")

                            if "virtue" in cat.lower():
                                sentiment.append("pos")
                            elif "vice" in cat.lower():
                                sentiment.append("neg")
                            else:
                                sentiment.append("neu")

                            if "Harm" in cat or "care" in cat:
                                moral_type.append("harm")
                            elif "Fairness" in cat or "fairness" in cat:
                                moral_type.append("fairness")
                            elif "Ingroup" in cat or "loyalty" in cat:
                                moral_type.append("loyalty")
                            elif "Authority" in cat or "authority" in cat:
                                moral_type.append("authority")
                            elif "Purity" in cat or "sanctity" in cat:
                                moral_type.append("purity")
                            else:
                                moral_type.append("other")
        print("parent")
        for age in parent_dict[corpus]:
            print(age)
            if age == None:
                continue
            year, month, _ = age
            year_frac = year + (month / 12)
            for utterance in parent_dict[corpus][age]:



                if with_pos:
                    parent_no_punct = [item for item in utterance if item[0] not in string.punctuation]
                else:
                    parent_no_punct = [item for item in utterance if item not in string.punctuation]


                for item in parent_no_punct:
                    if with_pos:
                        word = item[0]
                    else:
                        word = item


                    category = None
                    if word in pure_categories:
                        category = pure_categories[word]
                    elif any(key[:-1] in word for key in wildcard_categories):
                        for key in wildcard_categories:
                            if key[:-1] in item:
                                category = wildcard_categories[key]
                    if category == None:
                        years.append(year_frac)
                        corpora.append(corpus)
                        if with_pos:
                            words.append(item[0])
                            pos.append(item[1])
                        else:
                            words.append(item)

                        speech_categories.append(None)
                        identity.append("parent")
                        child_gender.append(item[2])
                        gender.append(item[3])
                        race.append(item[4])
                        family_class.append(item[5])
                        sentiment.append(None)
                        moral_type.append(None)
                    else:
                        for cat in category:
                            years.append(year_frac)
                            corpora.append(corpus)
                            if with_pos:
                                words.append(item[0])
                                pos.append(item[1])
                            else:
                                words.append(item)

                            speech_categories.append(cat)
                            identity.append("parent")
                            child_gender.append(item[2])
                            gender.append(item[3])
                            race.append(item[4])
                            family_class.append(item[5])

                            if "virtue" in cat.lower():
                                sentiment.append("pos")
                            elif "vice" in cat.lower():
                                sentiment.append("neg")
                            else:
                                sentiment.append("neu")

                            if "Harm" in cat or "care" in cat:
                                moral_type.append("harm")
                            elif "Fairness" in cat or "fairness" in cat:
                                moral_type.append("fairness")
                            elif "Ingroup" in cat or "loyalty" in cat:
                                moral_type.append("loyalty")
                            elif "Authority" in cat or "authority" in cat:
                                moral_type.append("authority")
                            elif "Purity" in cat or "sanctity" in cat:
                                moral_type.append("purity")
                            else:
                                moral_type.append("other")
    if with_pos:
        cols = {"year": years, "identity": identity, "words": words, "pos": pos,
                "category": speech_categories, "sentiment": sentiment, "type": moral_type, "corpus": corpora,  "race": race, "child gender": child_gender, "gender": gender, "class": family_class}
    else:
        cols = {"year": years, "identity": identity, "words": words,  "category": speech_categories,
                "sentiment": sentiment, "type": moral_type, "corpus": corpora, "race": race, "child gender": child_gender, "gender": gender, "class": family_class}


    return pd.DataFrame(cols)


def moral_categories_over_time_by_utterance(parent_child_dict: dict, pure_categories: dict, wildcard_categories: dict,
                                            with_pos: bool = True) -> pd.DataFrame:
    years = []
    corpora = []
    morality_keywords = []
    speech_categories = []
    identity = []
    sentiment = []
    moral_type = []
    context = []
    child_gender = []
    race = []
    family_class = []
    gender = []

    #
    # (parent_word, parent_pos, child_gender, 'male', race, family_class))

    for corpus in parent_child_dict["parent"]:
        parent_dict = parent_child_dict["parent"]
        child_dict = parent_child_dict["child"]
        print(corpus)
        print("child")
        for age in child_dict[corpus]:
            print(age)
            year, month, _ = age
            year_frac = year + (month / 12)

            for utterance in child_dict[corpus][age]:

                utterance_words = " ".join([item[0] for item in utterance])

                if with_pos:
                    child_no_punct = [item for item in utterance if item[0] not in string.punctuation]
                else:
                    child_no_punct = [item for item in utterance if item not in string.punctuation]

                category = []
                moral_words = []
                for item in child_no_punct:
                    if with_pos:
                        word = item[0]
                    else:
                        word = item
                    if word in pure_categories:
                        category.append(pure_categories[word])
                        moral_words.append([word])
                    elif any(key[:-1] in word for key in wildcard_categories):
                        for key in wildcard_categories:
                            if key[:-1] in word:
                                category.append(wildcard_categories[key])
                                moral_words.append([word])

                item = child_no_punct[0]
                print(item)
                if category == []:
                    years.append(year_frac)
                    corpora.append(corpus)
                    child_gender.append(item[2])
                    gender.append(item[3])
                    race.append(item[4])
                    family_class.append(item[5])
                    speech_categories.append(None)
                    identity.append("child")
                    sentiment.append(None)
                    moral_type.append(None)
                    context.append(utterance_words)
                    morality_keywords.append(None)

                else:
                    for i, cat_group in enumerate(category):
                        for cat in cat_group:
                            years.append(year_frac)
                            corpora.append(corpus)
                            child_gender.append(item[2])
                            gender.append(item[3])
                            race.append(item[4])
                            family_class.append(item[5])
                            speech_categories.append(cat)
                            identity.append("child")
                            context.append(utterance_words)
                            morality_keywords.append(moral_words[i][0])

                            if "virtue" in cat.lower():
                                sentiment.append("pos")
                            elif "vice" in cat.lower():
                                sentiment.append("neg")
                            else:
                                sentiment.append("neu")

                            if "Harm" in cat or "care" in cat:
                                moral_type.append("harm")
                            elif "Fairness" in cat or "fairness" in cat:
                                moral_type.append("fairness")
                            elif "Ingroup" in cat or "loyalty" in cat:
                                moral_type.append("loyalty")
                            elif "Authority" in cat or "authority" in cat:
                                moral_type.append("authority")
                            elif "Purity" in cat or "sanctity" in cat:
                                moral_type.append("purity")
                            else:
                                moral_type.append("other")
        print("parent")
        for age in parent_dict[corpus]:
            print(age)
            if age == None:
                continue
            year, month, _ = age
            year_frac = year + (month / 12)

            for utterance in parent_dict[corpus][age]:
                utterance_words = " ".join([item[0] for item in utterance])
                if with_pos:
                    parent_no_punct = [item for item in utterance if item[0] not in string.punctuation]
                else:
                    parent_no_punct = [item for item in utterance if item not in string.punctuation]

                category = []
                moral_words = []
                for item in parent_no_punct:
                    if with_pos:
                        word = item[0]
                    else:
                        word = item
                    if word in pure_categories:
                        category.append(pure_categories[word])
                        moral_words.append([word])
                    elif any(key[:-1] in word for key in wildcard_categories):
                        for key in wildcard_categories:
                            if key[:-1] in item:
                                category.append(wildcard_categories[key])
                                moral_words.append([word])
                # if len(category) != 0:
                # pdb.set_trace()
                item = parent_no_punct[0]
                if category == []:
                    years.append(year_frac)
                    corpora.append(corpus)
                    child_gender.append(item[2])
                    gender.append(item[3])
                    race.append(item[4])
                    family_class.append(item[5])
                    speech_categories.append(None)
                    identity.append("parent")
                    sentiment.append(None)
                    moral_type.append(None)
                    context.append(utterance_words)
                    morality_keywords.append(None)
                else:
                    for i, cat_group in enumerate(category):
                        for cat in cat_group:
                            years.append(year_frac)
                            corpora.append(corpus)
                            child_gender.append(item[2])
                            gender.append(item[3])
                            race.append(item[4])
                            family_class.append(item[5])
                            speech_categories.append(cat)
                            identity.append("parent")
                            context.append(utterance_words)
                            morality_keywords.append(moral_words[i][0])

                            if "virtue" in cat.lower():
                                sentiment.append("pos")
                            elif "vice" in cat.lower():
                                sentiment.append("neg")
                            else:
                                sentiment.append("neu")

                            if "Harm" in cat or "care" in cat:
                                moral_type.append("harm")
                            elif "Fairness" in cat or "fairness" in cat:
                                moral_type.append("fairness")
                            elif "Ingroup" in cat or "loyalty" in cat:
                                moral_type.append("loyalty")
                            elif "Authority" in cat or "authority" in cat:
                                moral_type.append("authority")
                            elif "Purity" in cat or "sanctity" in cat:
                                moral_type.append("purity")
                            else:
                                moral_type.append("other")

    cols = {"year": years, "identity": identity, "category": speech_categories, "sentiment": sentiment,
            "type": moral_type, "corpus": corpora, "context": context, "keywords": morality_keywords, "race": race, "child gender": child_gender, "gender": gender, "class": family_class}

    return pd.DataFrame(cols)


def extract_modal_sentences(corpora_list: List, out_filename: str = "./data/modal_sentences.txt") -> List:
    """Extracts all sentences containing modal verbs from specified corpora.

    Args:
        corpora_list (List): list of corpora

    Returns:
        List: [[modal sentence],....]
    """
    modal_sentences = []
    modal_verbs = ["can", "cannot", "could", "must", "need to", "may", "shall", "have to", "ought to", "should",
                   "allow", "force", "would"]

    for corpus in corpora_list:
        print(corpus)
        url = f"{base_url}/{corpus}"
        chats = pylangacq.read_chat(url)
        ages = chats.ages()

        words_by_files_chi = chats.utterances(participants="CHI", by_files=True)
        words_by_files_pa = chats.utterances(participants={"MOT", "FAT"}, by_files=True)

        for age, words_child, words_parents in zip(ages, words_by_files_chi, words_by_files_pa):
            for utterance in words_child:
                sentence = " ".join([token.word for token in utterance.tokens])
                if any(modal_verb in sentence for modal_verb in modal_verbs):
                    modal_sentences.append(["CHILD: " + sentence])

            for utterance in words_parents:
                sentence = " ".join([token.word for token in utterance.tokens])
                if any(modal_verb in sentence for modal_verb in modal_verbs):
                    modal_sentences.append(["PARENT: " + sentence])

    with open(out_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(modal_sentences)

    return modal_sentences


def extract_category_sentences(corpora_list: List, category_names: List, pure_cats: dict, wildcard_cats: dict = {},
                               out_dir: str = "./data/", size: int = 100) -> None:
    category_sentences_chi = {cat_name: [] for cat_name in category_names}
    category_sentences_pa = {cat_name: [] for cat_name in category_names}

    for corpus in corpora_list:
        print(corpus)
        url = f"{base_url}/{corpus}"
        chats = pylangacq.read_chat(url)
        ages = chats.ages()

        words_by_files_chi = chats.utterances(participants="CHI", by_files=True)
        words_by_files_pa = chats.utterances(participants={"MOT", "FAT"}, by_files=True)

        for age, words_child, words_parents in zip(ages, words_by_files_chi, words_by_files_pa):
            for utterance in words_child:
                list_words = [token.word for token in utterance.tokens]
                sentence = " ".join(list_words)
                done = False

                for word in list_words:
                    if done:
                        break
                    if word in pure_cats:
                        categories = pure_cats[word]
                        for category in categories:
                            category_sentences_chi[category].append(["CHILD: " + sentence])
                        done = True
                    for match in wildcard_cats:
                        if match[:-1] in word:
                            try:
                                categories = wildcard_cats[match]
                            except:
                                pdb.set_trace()
                            for category in categories:
                                category_sentences_chi[category].append(["CHILD: " + sentence])
                            done = True
                            break

            for utterance in words_parents:
                list_words = [token.word for token in utterance.tokens]
                sentence = " ".join(list_words)
                done = False

                for word in list_words:
                    if word in pure_cats:
                        categories = pure_cats[word]
                        for category in categories:
                            category_sentences_pa[category].append(["PARENT: " + sentence])
                        done = True
                    for match in wildcard_cats:
                        if match[:-1] in word:
                            categories = wildcard_cats[match]
                            for category in categories:
                                category_sentences_pa[category].append(["PARENT: " + sentence])
                            done = True
                            break

    for category in category_sentences_pa:
        shuffle(category_sentences_chi[category])
        child_sample = category_sentences_chi[category][:size]
        shuffle(category_sentences_pa[category])
        parent_sample = category_sentences_pa[category][:size]

        child_out_path = os.path.join(out_dir, f"child-{category}.csv")
        parent_out_path = os.path.join(out_dir, f"parent-{category}.csv")

        with open(child_out_path, "w") as ch_out:
            writer = csv.writer(ch_out)
            writer.writerows(child_sample)

        with open(parent_out_path, "w") as pa_out:
            writer = csv.writer(pa_out)
            writer.writerows(parent_sample)


if __name__ == "__main__":



    base_url = "https://childes.talkbank.org/data/Eng-NA/"
    corpora = corpora_eng


    read_all_corpora_by_utterance(corpora,f'data/childes-dict-all-lemmatized.p', append_pos=False, lemmatize=True)
    childes_data = pickle.load(open(f'data/childes-dict-all-lemmatized.p', 'rb'))



    categories_v2 = read_mfd("./data/mfd2.0.dic", version=2)
    wildcard_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] == "*"}
    pure_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] != "*"}

    categories_mfd_2 = ["care.virtue", "care.vice", "fairness.virtue", "fairness.vice", "loyalty.virtue",
                        "loyalty.vice", "authority.virtue", "authority.vice", "sanctity.virtue", "sanctity.vice"]

    moral_df = moral_categories_over_time(childes_data, with_pos=True,
                                                                  pure_categories=pure_categories_v2,
                                                                  wildcard_categories=wildcard_categories_v2)
    #
    moral_df.to_pickle(f"./data/moral_df_all-lemmatized.p")
    #
    moral_df_utterances = moral_categories_over_time_by_utterance(childes_data, with_pos=True,
                                                       pure_categories=pure_categories_v2,
                                                       wildcard_categories=wildcard_categories_v2)
    moral_df_utterances.to_pickle(f"./data/moral_df_context_all-lemmatized.p")





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

corpora = [
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

positive_cats = ["HarmVirtue", "FairnessVirtue", "IngroupVirtue", "AuthorityVirtue", "PurityVirtue", "care.virtue", "fairness.virtue",
"loyalty.virtue", "authority.virtue", "sanctity.virtue"]
negative_cats = ["HarmVice", "FairnessVice", "IngroupVice", "AuthorityVice", "PurityVice", "care.vice", "fairness.vice",
"loyalty.vice", "authority.vice", "sanctity.vice"]

base_url = "https://childes.talkbank.org/data/Eng-NA/"

def read_all_corpora(corpora_list: List, out_filename: str = "data/childes-dict.p", append_pos: bool = False, lemmatize: bool = True) -> tuple:
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
        #pdb.set_trace()
        if append_pos:
            tokens_by_files_chi = chats.tokens(participants="CHI", by_files=True)
            tokens_by_files_mother = chats.tokens(participants="MOT", by_files=True)
            tokens_by_files_father = chats.tokens(participants="FAT", by_files=True)

            corpora_child[corpus[:-4]] = {}
            corpora_parents[corpus[:-4]] = {}

            for age, header, tokens_child, tokens_mother, tokens_father in zip(ages, headers, tokens_by_files_chi, tokens_by_files_mother, tokens_by_files_father):
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

def find_mfd_childes_intersection(parent_child_dict: dict, mode: str = "percent") -> float:
    """How much overlap is there between the words in CHILDES and what's in the moral foundations dictionary?

    Args:
        parent_child_dict (dict): same format as output of read_all_corpora
        mode (str): should be ["percent", "number"]. Whether to return the percentage of words intersecting (from CHILDES) or the absolute number.

    Returns:
        float: Either the percentage of words intersecting or the absolute number of intersections.
    """
    total_words = 0
    moral_words = 0
    moral_words_list = []
    for corpus in parent_child_dict["parent"]:
        parent_dict = parent_child_dict["parent"]
        child_dict = parent_child_dict["child"]
        for age in child_dict[corpus]:
            child_no_punct = [item for item in child_dict[corpus][age] if item not in string.punctuation]
            parent_no_punct = [item for item in parent_dict[corpus][age] if item not in string.punctuation]
            moral_words_child = []
            moral_words_parent = []

            for item in child_no_punct:
                if item in pure_categories or any(key[:-1] in item for key in wildcard_categories):
                    moral_words_child.append(item)
                    moral_words_list.append(item)
            for item in parent_no_punct:
                if item in pure_categories or any(key[:-1] in item for key in wildcard_categories):
                    moral_words_parent.append(item)
                    moral_words_list.append(item)

            total_words += len(child_no_punct)
            total_words += len(parent_no_punct)
            moral_words += len(moral_words_child)
            moral_words += len(moral_words_parent)

    if mode == "percent": 
        print(moral_words_list)
        pdb.set_trace()
        return moral_words/total_words
    return moral_words

def moral_categories_over_time(parent_child_dict: dict, pure_categories: dict, wildcard_categories: dict, with_pos: bool = False) -> pd.DataFrame:
    years = []
    corpora = []
    words = []
    pos = []
    gender = []
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
            #if corpus == "Bliss" and age == (5, 4, 0):
                #pdb.set_trace()
            year, month, _ = age
            year_frac = year + (month/12)
            if with_pos:
                child_no_punct = [item for item in child_dict[corpus][age] if item[0] not in string.punctuation]
            else:
                child_no_punct = [item for item in child_dict[corpus][age] if item not in string.punctuation]

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
                    years.append(year_frac)
                    corpora.append(corpus)
                    if with_pos:
                        words.append(item[0])
                        pos.append(item[1])
                    else:
                        words.append(item)
                    gender.append(item[2])
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
                        gender.append(item[2])
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
            year_frac = year + (month/12)

            if with_pos:
                parent_no_punct = [item for item in parent_dict[corpus][age] if item[0] not in string.punctuation]
            else:
                parent_no_punct = [item for item in parent_dict[corpus][age] if item not in string.punctuation]

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
                    gender.append(item[2])
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
                        gender.append(item[2])

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
        cols = {"year": years, "identity": identity, "words": words, "pos": pos,"gender": gender, "category": speech_categories, "sentiment": sentiment, "type": moral_type, "corpus": corpora}
    else:
        cols = {"year": years, "identity": identity, "words": words,"gender": gender, "category": speech_categories, "sentiment": sentiment, "type": moral_type, "corpus": corpora}
    return pd.DataFrame(cols)

def extract_modal_sentences(corpora_list: List, out_filename: str = "./data/modal_sentences.txt") -> List:
    """Extracts all sentences containing modal verbs from specified corpora.

    Args:
        corpora_list (List): list of corpora

    Returns:
        List: [[modal sentence],....]
    """
    modal_sentences = []
    modal_verbs = ["can", "cannot", "could", "must", "need to", "may", "shall", "have to", "ought to", "should", "allow", "force", "would"]

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

def extract_category_sentences(corpora_list: List, category_names: List, pure_cats: dict, wildcard_cats: dict = {}, out_dir: str = "./data/", size: int = 100) -> None:
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
    #read_all_corpora(corpora, out_filename="./data/childes-dict-lemmatized.p", append_pos=True, lemmatize=True)
    #assert False

    categories = read_mfd("./data/mfd.dic")
    wildcard_categories = {cat: val for cat, val in categories.items() if cat[-1] == "*"}
    pure_categories = {cat: val for cat, val in categories.items() if cat[-1] != "*"}
    categories_v2 = read_mfd("./data/mfd2.0.dic", version=2)
    wildcard_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] == "*"}
    pure_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] != "*"}
    categories_mfd_1 = ["HarmVirtue", "HarmVice", "FairnessVirtue", "FairnessVice", "IngroupVirtue", "IngroupVice", "AuthorityVirtue", "AuthorityVice", "PurityVirtue", "PurityVice", "MoralityGeneral"]
    categories_mfd_2 = ["care.virtue", "care.vice", "fairness.virtue", "fairness.vice", "loyalty.virtue", "loyalty.vice", "authority.virtue", "authority.vice", "sanctity.virtue", "sanctity.vice"]
    #extract_category_sentences(corpora, categories_mfd_2, pure_categories_v2, wildcard_categories_v2, "./data/sample-sentences/")
    #assert False
    categories_v2 = read_mfd("./data/mfd2.0.dic", version=2)
    wildcard_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] == "*"}
    pure_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] != "*"}
    parent_child_dict = pickle.load(open("./data/childes-dict-lemmatized.p", "rb"))
    moral_df = moral_categories_over_time(parent_child_dict, with_pos=True, pure_categories=pure_categories, wildcard_categories=wildcard_categories)
    moral_df.to_pickle("./data/moral_df_lemma.p")
    #find_mfd_childes_intersection(parent_child_dict)

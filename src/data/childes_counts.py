from collections import Counter
import string
from typing import List

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

categories = read_mfd("./data/mfd.dic")
wildcard_categories = {cat: val for cat, val in categories.items() if cat[-1] == "*"}
pure_categories = {cat: val for cat, val in categories.items() if cat[-1] != "*"}

positive_cats = ["HarmVirtue", "FairnessVirtue", "IngroupVirtue", "AuthorityVirtue", "PurityVirtue"]
negative_cats = ["HarmVice", "FairnessVice", "IngroupVice", "AuthorityVice", "PurityVice"]

base_url = "https://childes.talkbank.org/data/Eng-NA/"

def read_all_corpora(corpora_list: List, out_filename: str = "data/childes-dict.p", append_pos: bool = False) -> tuple:
    """Reads a specified list of corpora from CHILDES and returns data tagged by age.

    Args:
        corpora_list (List): List of corpora names (default listed above)
        out_filename (str): Location to write out final dict
        append_pos (bool): whether to store just the words in an utterance or words and POS
    Returns:
        tuple[dict]: {corpus name: {age: [...word/(word, POS)...]}} for children and parents
    """
    corpora_child = {}
    corpora_parents = {}

    for corpus in corpora_list:
        print(corpus)
        url = f"{base_url}/{corpus}"
        chats = pylangacq.read_chat(url)
        ages = chats.ages()

        if append_pos:
            tokens_by_files_chi = chats.tokens(participants="CHI", by_files=True)
            tokens_by_files_pa = chats.tokens(participants={"MOT", "FAT"}, by_files=True)

            corpora_child[corpus[:-4]] = {}
            corpora_parents[corpus[:-4]] = {}

            for age, tokens_child, tokens_parents in zip(ages, tokens_by_files_chi, tokens_by_files_pa):
                if age in corpora_child[corpus[:-4]]:
                    for item_c, item_p in zip(tokens_child, tokens_parents):
                        child_word, child_pos = item_c.word, item_c.pos
                        parent_word, parent_pos = item_p.word, item_p.pos
                        corpora_child[corpus[:-4]][age].append((child_word, child_pos))
                        corpora_parents[corpus[:-4]][age].append((parent_word, parent_pos))
                else:
                    for item_c, item_p in zip(tokens_child, tokens_parents):
                        child_word, child_pos = item_c.word, item_c.pos
                        parent_word, parent_pos = item_p.word, item_p.pos
                        corpora_child[corpus[:-4]][age] = [(child_word, child_pos)]
                        corpora_parents[corpus[:-4]][age] = [(parent_word, parent_pos)]
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

def moral_categories_over_time(parent_child_dict: dict, with_pos: bool = False) -> pd.DataFrame:
    years = []
    words = []
    pos = []
    speech_categories = []
    identity = []
    sentiment = []
    moral_type = []

    for corpus in parent_child_dict["parent"]:
        parent_dict = parent_child_dict["parent"]
        child_dict = parent_child_dict["child"]
        for age in child_dict[corpus]:
            year, month, _ = age
            year_frac = year + (month/12)
            if with_pos:
                child_no_punct = [item for item in child_dict[corpus][age] if item[0] not in string.punctuation]
                parent_no_punct = [item for item in parent_dict[corpus][age] if item[0] not in string.punctuation]
            else:
                child_no_punct = [item for item in child_dict[corpus][age] if item not in string.punctuation]
                parent_no_punct = [item for item in parent_dict[corpus][age] if item not in string.punctuation]

            for item in child_no_punct:
                category = None
                if item in pure_categories:
                    category = categories[item]
                elif any(key[:-1] in item for key in wildcard_categories):
                    for key in wildcard_categories:
                        if key[:-1] in item:
                            category = categories[key]
                if category == None:
                    years.append(year_frac)
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
                        if with_pos:
                            words.append(item[0])
                            pos.append(item[1])
                        else:
                            words.append(item)
                        speech_categories.append(cat)
                        identity.append("child")

                        if "Virtue" in cat:
                            sentiment.append("pos")
                        elif "Vice" in cat:
                            sentiment.append("neg")
                        else:
                            sentiment.append("neu")
                        
                        if "Harm" in cat:
                            moral_type.append("harm")
                        elif "Fairness" in cat:
                            moral_type.append("fairness")
                        elif "Ingroup" in cat:
                            moral_type.append("loyalty")
                        elif "Authority" in cat:
                            moral_type.append("authority")
                        elif "Purity" in cat:
                            moral_type.append("purity")
                        else:
                            moral_type.append("other")
            
            for item in parent_no_punct:
                category = None
                if item in pure_categories:
                    category = categories[item]
                elif any(key[:-1] in item for key in wildcard_categories):
                    for key in wildcard_categories:
                        if key[:-1] in item:
                            category = categories[key]
                if category == None:
                    years.append(year_frac)
                    if with_pos:
                        words.append(item[0])
                        pos.append(item[1])
                    else:
                        words.append(item)
                    speech_categories.append(None)
                    identity.append("parent")
                    sentiment.append(None)
                    moral_type.append(None)
                else:
                    for cat in category:
                        years.append(year_frac)
                        if with_pos:
                            words.append(item[0])
                            pos.append(item[1])
                        else:
                            words.append(item)
                        speech_categories.append(cat)
                        identity.append("parent")

                        if "Virtue" in cat:
                            sentiment.append("pos")
                        elif "Vice" in cat:
                            sentiment.append("neg")
                        else:
                            sentiment.append("neu")
                        
                        if "Harm" in cat:
                            moral_type.append("harm")
                        elif "Fairness" in cat:
                            moral_type.append("fairness")
                        elif "Ingroup" in cat:
                            moral_type.append("loyalty")
                        elif "Authority" in cat:
                            moral_type.append("authority")
                        elif "Purity" in cat:
                            moral_type.append("purity")
                        else:
                            moral_type.append("other")
        if with_pos:
            cols = {"year": years, "identity": identity, "words": words, "pos": pos, "category": speech_categories, "sentiment": sentiment, "type": moral_type}
        else:
            cols = {"year": years, "identity": identity, "words": words, "category": speech_categories, "sentiment": sentiment, "type": moral_type}
        return pd.DataFrame(cols)


if __name__ == "__main__":
    #read_all_corpora(corpora, out_filename="./data/childes-dict-pos.p", append_pos=True)
    parent_child_dict = pickle.load(open("./data/childes-dict.p", "rb"))
    moral_categories_over_time(parent_child_dict)
    #find_mfd_childes_intersection(parent_child_dict)

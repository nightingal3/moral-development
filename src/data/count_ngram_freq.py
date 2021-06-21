import pandas as pd
import pdb
import pickle
from typing import List
from wordfreq import word_frequency

from src.data.mfd_read import read_mfd

def count_frequencies(mfd_dict: dict, out_path: str) -> pd.DataFrame:
    words = []
    freqs = []
    for word in mfd_dict:
        freq = word_frequency(word, "en")
        words.append(word)
        freqs.append(freq)

    cols = {"word": words, "frequency": freqs}
    out_df = pd.DataFrame(cols)
    out_df.to_csv(out_path, index=False)

    return out_df

def count_frequencies_childes(childes_df: pd.DataFrame, mfd_dict: dict, out_path: str) -> pd.DataFrame:
    words = []
    freqs = []
    val_counts = childes_df["words"].value_counts().to_dict()

    for word in mfd_dict:
        if word not in val_counts:
            freq = 0
        else:
            freq = val_counts[word]/len(childes_df)
        words.append(word)
        freqs.append(freq)

    cols = {"word": words, "frequency": freqs}
    out_df = pd.DataFrame(cols)
    out_df.to_csv(out_path, index=False)

    return out_df


def get_top_k_words_per_value(freq_df: pd.DataFrame, mfd_dict: dict, cat_names: List, k: int, return_list: bool = False) -> dict:
    top_k_values = {}
    for value in cat_names.values():
        word_freqs = []
        mfd_subset = [word for word, word_value in mfd_dict.items() if value in word_value]
        for word in mfd_subset:
            freq = freq_df.query(f"word == '{word}'")["frequency"].item()
            word_freqs.append((word, freq))
    
        top_k = sorted(word_freqs, key=lambda x: x[1], reverse=True)[:k]
        top_k_values[value] = top_k

    if return_list:
        values_list = []
        for value in top_k_values:
            values_list.extend([item[0] for item in top_k_values[value]])

        return values_list
        
    return top_k_values


if __name__ == "__main__":
    categories_v2, cat_names = read_mfd("./data/mfd2.0.dic", version=2, return_categories=True)
    wildcard_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] == "*"}
    pure_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] != "*"}

    childes_df = pickle.load(open("./data/moral_df_2.p", "rb"))

    freq_df = count_frequencies(categories_v2, "./data/mfd-2-freq.csv")

    freq_childes = count_frequencies_childes(childes_df, categories_v2, "./data/mfd-2-childes-freq.csv")

    print(get_top_k_words_per_value(freq_childes, categories_v2, cat_names, 5))


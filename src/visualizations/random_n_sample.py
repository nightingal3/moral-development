from random import sample
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from typing import Callable

from src.visualizations.moral_categories import words_to_categories
from src.data.mfd_read import read_mfd

def sample_random_n_words(moral_dict: dict, n: int) -> dict:
    new_dict = {}
    for value in moral_dict:
        new_dict[value] = sample(moral_dict[value], n)
    return new_dict

def format_dict(words_to_values: dict) -> dict:
    values_to_words = {}
    for word in words_to_values:
        for value in words_to_values[word]:
            if value not in values_to_words:
                values_to_words[value] = [word]
            else:
                values_to_words[value].append(word)

    return values_to_words

def make_relabel_fn(new_values_dict: dict) -> Callable:
    def relabel(row):
        for value in new_values_dict:
            if row["words"] in new_values_dict[value]:
                return value
            return None

    return relabel
    

def get_time_trajectory(categories_df: pd.DataFrame, person: str, moral_dict: dict, num_samples: int, n: int) -> pd.DataFrame:
    all_data = pd.DataFrame()
    for _ in range(num_samples):
        copy_df = categories_df.copy()
        sampled_dict = sample_random_n_words(moral_dict, n)
        relabel_fn = make_relabel_fn(sampled_dict)
        #pdb.set_trace()
        copy_df["old_type"] = copy_df["type"]
        copy_df["type"] = copy_df.apply(relabel_fn, axis=1)
        trial_data = words_to_categories(copy_df, person, split_type="orig", exclude_hall=True, return_values_only=True)

        all_data = pd.concat([all_data, trial_data], ignore_index=True)

    return all_data

def plot_data(all_data: pd.DataFrame, out_filename: str) -> None:
    sns.lineplot(data=all_data, x="year", y="percentage", hue="moral_type")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.xticks([i for i in range(8)])
    plt.savefig(out_filename)

if __name__ == "__main__":
    categories_v2, cat_names = read_mfd("./data/mfd2.0.dic", version=2, return_categories=True)
    wildcard_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] == "*"}
    pure_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] != "*"}

    categories_df = pickle.load(open("./data/moral_df_2.p", "rb"))

    moral_dict = format_dict(categories_v2)

    all_df = get_time_trajectory(categories_df, "child", moral_dict, 30, 10)
    all_df.to_csv(".data/all-df.csv")
    plot_data(all_df, "random-n-child.png")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import pickle
from typing import List

from src.data.childes_counts import moral_categories_over_time
from src.data.mfd_read import read_mfd
from src.data.count_ngram_freq import get_top_k_words_per_value, count_frequencies_childes

def words_to_categories(categories_df: pd.DataFrame, person: str = "child", split_type: str = "sentiment", exclude_hall: bool = False, words_exclude: List = [], use_frequency_weighting: bool = False, return_values_only: bool = False) -> None:
    if words_exclude != []:
        mask = categories_df['words'].isin(words_exclude)
        categories_df = categories_df[~mask]
    if exclude_hall:
        categories_df = categories_df.loc[categories_df["corpus"] != "Hall"]
    
    if use_frequency_weighting:
        word_freqs = pd.read_csv("./data/mfd-2-freq.csv")

    if split_type == "sentiment":
        harm_virtue = categories_df.query(f'((type == "harm") & (identity == "{person}")) & (sentiment == "pos")')
        harm_vice = categories_df.query(f'((type == "harm") & (identity == "{person}")) & (sentiment == "neg")')
        fair_virtue = categories_df.query(f'((type == "fairness") & (identity == "{person}")) & (sentiment == "pos")')
        fair_vice = categories_df.query(f'((type == "fairness") & (identity == "{person}")) & (sentiment == "neg")')
        loyal_virtue = categories_df.query(f'((type == "loyalty") & (identity == "{person}")) & (sentiment == "pos")')
        loyal_vice = categories_df.query(f'((type == "loyalty") & (identity == "{person}")) & (sentiment == "neg")')
        authority_virtue = categories_df.query(f'((type == "authority") & (identity == "{person}")) & (sentiment == "pos")')
        authority_vice = categories_df.query(f'((type == "authority") & (identity == "{person}")) & (sentiment == "neg")')
        pure_virtue = categories_df.query(f'((type == "purity") & (identity == "{person}")) & (sentiment == "pos")')
        pure_vice = categories_df.query(f'((type == "purity") & (identity == "{person}")) & (sentiment == "neg")')        
    else:
        harm = categories_df.query(f'(type == "harm") & (identity == "{person}")')
        fair = categories_df.query(f'(type == "fairness") & (identity == "{person}")')
        loyal = categories_df.query(f'(type == "loyalty") & (identity == "{person}")')
        authority = categories_df.query(f'(type == "authority") & (identity == "{person}")')
        pure = categories_df.query(f'(type == "purity") & (identity == "{person}")')

        values_dict = {"harm": harm, "fair": fair, "loyal": loyal, "authority": authority, "pure": pure}

    years = []
    percentage = []
    moral_type = []

    years_care = []
    percent_care = []
    years_harm = []
    percent_harm = []
    years_fair = []
    percent_fair = []
    years_unfair = []
    percent_unfair = []
    years_loyal = []
    percent_loyal = []
    years_betray = []
    percent_betray = []
    years_loyal = []
    years_authority = []
    percent_authority = []
    years_subversion = []
    percent_subversion = []
    years_pure = []
    percent_pure = []
    years_degrade = []
    percent_degrade = []

    years_pos = []
    pos_values = {"harm": {}, "fair": {}, "loyal": {}, "authority": {}, "pure": {}}

    for year in range(int(min(categories_df.year)), int(max(categories_df.year)) + 1):
        words_year = categories_df.loc[(categories_df["year"] > year) & (categories_df["year"] <= year + 1)]
        unique_words = words_year.dropna(subset=["category"])
        unique_words_count_harm = unique_words.query(f"(type == 'harm') & (identity == '{person}')")["words"].value_counts().rename_axis('unique_values').reset_index(name='counts')
        unique_words_count_fair = unique_words.query(f"type == 'fair' & (identity == '{person}')")["words"].value_counts().rename_axis('unique_values').reset_index(name='counts')
        unique_words_count_loyal = unique_words.query(f"type == 'loyal' & (identity == '{person}')")["words"].value_counts().rename_axis('unique_values').reset_index(name='counts')
        unique_words_count_authority = unique_words.query(f"type == 'authority' & (identity == '{person}')")["words"].value_counts().rename_axis('unique_values').reset_index(name='counts')
        unique_words_count_purity = unique_words.query(f"type == 'purity' & (identity == '{person}')")["words"].value_counts().rename_axis('unique_values').reset_index(name='counts')

        if len(words_year) == 0:
            continue
        if split_type == "sentiment":
            years_care.append(year)
            years_harm.append(year)
            years_fair.append(year)
            years_unfair.append(year)
            years_loyal.append(year)
            years_betray.append(year)
            years_authority.append(year)
            years_subversion.append(year)
            years_pure.append(year)
            years_degrade.append(year)

            percentage_harm_pos = len(harm_virtue.loc[(harm_virtue["year"] > year) & (harm_virtue["year"] <= year + 1)])/len(words_year)
            percentage_harm_neg = len(harm_vice.loc[(harm_vice["year"] > year) & (harm_vice["year"] <= year + 1)])/len(words_year)
            percentage_fair_pos = len(fair_virtue.loc[(fair_virtue["year"] > year) & (fair_virtue["year"] <= year + 1)])/len(words_year)
            percentage_fair_neg = len(fair_vice.loc[(fair_vice["year"] > year) & (fair_vice["year"] <= year + 1)])/len(words_year)
            percentage_loyal_pos = len(loyal_virtue.loc[(loyal_virtue["year"] > year) & (loyal_virtue["year"] <= year + 1)])/len(words_year)
            percentage_loyal_neg = len(loyal_vice.loc[(loyal_vice["year"] > year) & (loyal_vice["year"] <= year + 1)])/len(words_year)
            percentage_authority_pos = len(authority_virtue.loc[(authority_virtue["year"] > year) & (authority_virtue["year"] <= year + 1)])/len(words_year)
            percentage_authority_neg = len(authority_vice.loc[(authority_vice["year"] > year) & (authority_vice["year"] <= year + 1)])/len(words_year)
            percentage_pure_pos = len(pure_virtue.loc[(pure_virtue["year"] > year) & (pure_virtue["year"] <= year + 1)])/len(words_year)
            percentage_pure_neg = len(pure_vice.loc[(pure_vice["year"] > year) & (pure_vice["year"] <= year + 1)])/len(words_year)

            percent_care.append(percentage_harm_pos)
            percent_harm.append(percentage_harm_neg)
            percent_fair.append(percentage_fair_pos)
            percent_unfair.append(percentage_fair_neg)
            percent_loyal.append(percentage_loyal_pos)
            percent_betray.append(percentage_loyal_neg)
            percent_authority.append(percentage_authority_pos)
            percent_subversion.append(percentage_authority_neg)
            percent_pure.append(percentage_pure_pos)
            percent_degrade.append(percentage_pure_neg)
        elif split_type == "pos":
            years_pos.append(year)
            for value in pos_values: 

                value_df = values_dict[value]
                df_year = value_df.loc[(value_df["year"] > year) & (value_df["year"] <= year + 1)]
                percentage_noun = len(df_year.loc[df_year["pos"] == "n"])/len(words_year)
                percentage_verb = len(df_year.loc[df_year["pos"] == "v"])/len(words_year)
                percentage_adj = len(df_year.loc[df_year["pos"] == "adj"])/len(words_year)

                if len(pos_values[value]) == 0:
                    pos_values[value]["noun"] = []
                    pos_values[value]["verb"] = []
                    pos_values[value]["adj"] = []

                pos_values[value]["noun"].append(percentage_noun)
                pos_values[value]["verb"].append(percentage_verb)
                pos_values[value]["adj"].append(percentage_adj)

        elif split_type == "gender":
            years.append(year)
            for value in pos_values: 
                value_df = values_dict[value]
                df_year = value_df.loc[(value_df["year"] > year) & (value_df["year"] <= year + 1)]
                percentage_male = len(df_year.loc[df_year["gender"] == "male"])/len(words_year)
                percentage_female = len(df_year.loc[df_year["gender"] == "female"])/len(words_year)

                if len(pos_values[value]) == 0:
                    pos_values[value]["male"] = []
                    pos_values[value]["female"] = []

                pos_values[value]["male"].append(percentage_male)
                pos_values[value]["female"].append(percentage_female)
        else:
            years.extend([year] * 5)

            
            percentage_harm = len(harm.loc[(harm["year"] > year) & (harm["year"] <= year + 1)])/len(words_year)
            percentage_fair = len(fair.loc[(fair["year"] > year) & (fair["year"] <= year + 1)])/len(words_year)
            percentage_loyal = len(loyal.loc[(loyal["year"] > year) & (loyal["year"] <= year + 1)])/len(words_year)
            percentage_authority = len(authority.loc[(authority["year"] > year) & (authority["year"] <= year + 1)])/len(words_year)
            percentage_pure = len(pure.loc[(pure["year"] > year) & (pure["year"] <= year + 1)])/len(words_year)

            if use_frequency_weighting:
                harm_combined = unique_words_count_harm.merge(word_freqs, how="left", left_on="unique_values", right_on="word")
                fair_combined = unique_words_count_fair.merge(word_freqs, how="left", left_on="unique_values", right_on="word")
                loyal_combined = unique_words_count_loyal.merge(word_freqs, how="left", left_on="unique_values", right_on="word")
                authority_combined = unique_words_count_authority.merge(word_freqs, how="left", left_on="unique_values", right_on="word")
                pure_combined = unique_words_count_purity.merge(word_freqs, how="left", left_on="unique_values", right_on="word")

                harm_combined["counts"] = harm_combined["counts"]/len(words_year)
                fair_combined["counts"] = fair_combined["counts"]/len(words_year)
                loyal_combined["counts"] = loyal_combined["counts"]/len(words_year)
                authority_combined["counts"] = authority_combined["counts"]/len(words_year)
                pure_combined["counts"] = pure_combined["counts"]/len(words_year)

                try:
                    percentage_harm = sum(harm_combined["counts"])/sum(harm_combined["frequency"])
                except:
                    percentage_harm = 0
                try:
                    percentage_fair = sum(fair_combined["counts"])/sum(fair_combined["frequency"])
                except:
                    percentage_fair = 0
                try:
                    percentage_loyal = sum(loyal_combined["counts"])/sum(loyal_combined["frequency"])
                except:
                    percentage_loyal = 0
                try:
                    percentage_authority = sum(authority_combined["counts"])/sum(authority_combined["frequency"])
                except:
                    percentage_authority = 0
                try:
                    percentage_pure = sum(pure_combined["counts"])/sum(pure_combined["frequency"])
                except:
                    percentage_pure = 0

            percentage.append(percentage_harm)
            moral_type.append("Harm")
            percentage.append(percentage_fair)
            moral_type.append("Fairness")
            percentage.append(percentage_loyal)
            moral_type.append("Loyalty")
            percentage.append(percentage_authority)
            moral_type.append("Authority")
            percentage.append(percentage_pure)
            moral_type.append("Purity")

    if split_type == "sentiment":
        plt.plot(years_care, percent_care, color="red", label="care")
        plt.plot(years_harm, percent_harm, color="red", linestyle="dashed", label="harm")
        plt.plot(years_fair, percent_fair, color="orange", label="fairness")
        plt.plot(years_unfair, percent_unfair, color="orange", linestyle="dashed", label="unfairness")
        plt.plot(years_loyal, percent_loyal, color="green", label="loyalty")
        plt.plot(years_betray, percent_betray, color="green", linestyle="dashed", label="betrayal")
        plt.plot(years_authority, percent_authority, color="blue", label="authority")
        plt.plot(years_subversion, percent_subversion, color="blue", linestyle="dashed", label="subversion")
        plt.plot(years_pure, percent_pure, color="purple", label="purity")
        plt.plot(years_degrade, percent_degrade, color="purple", linestyle="dashed", label="degradation")

        plt.xticks([i for i in range(8)])
        plt.xlabel("Year")
        plt.xlim(0, 8)
        plt.ylabel("Normalized Frequency")
        plt.legend()
        plt.savefig("parent-categories-split-nohall.png")
    elif split_type == "pos":
        styles = {"noun": "solid", "verb": "dashed", "adj": "dotted"}
        colors = {"harm": "red", "fair": "orange", "loyal": "green", "authority": "blue", "pure": "purple"}
        for value in pos_values:
            for pos in ["noun", "verb", "adj"]:
                plt.plot(years_pos, pos_values[value][pos], color=colors[value], linestyle=styles[pos], label=f"{value}-{pos}")
        
        plt.xticks([i for i in range(8)])
        plt.xlabel("Year")
        plt.xlim(0, 8)
        plt.ylabel("Normalized Frequency")
        plt.legend()
        plt.savefig("parent-pos-split-2.png")
    
    elif split_type == "gender":
        colors = {"harm": "red", "fair": "orange", "loyal": "green", "authority": "blue", "pure": "purple"}
        for value in pos_values:
            plt.plot(years, pos_values[value]["male"],color=colors[value], linestyle="solid", label=f"Male {value}")
            plt.plot(years, pos_values[value]["female"], color=colors[value], linestyle="dashed", label=f"Female {value}")

        plt.xticks([i for i in range(8)])
        plt.xlabel("Year")
        plt.xlim(0, 8)
        plt.ylabel("Normalized Frequency")
        plt.legend()
        plt.savefig("child-gender-split-1.png")

    else:
        cols_over_time = {"year": years, "percentage": percentage, "moral_type": moral_type}
        data_over_time = pd.DataFrame(cols_over_time)
        if return_values_only:
            return data_over_time

        sns.lineplot(data=data_over_time, x="year", y="percentage", hue="moral_type")
        plt.xticks([i for i in range(8)])
        plt.xlabel("Year")
        plt.xlim(0, 8)
        if use_frequency_weighting:
            plt.ylabel("Ratio of moral word frequency to expected moral word frequency")
        else:
            plt.ylabel("Normalized Frequency")
        plt.savefig("parent-frequency-corrected.png")

def num_words_by_age(categories_df: pd.DataFrame, out_filename: str = "words-per-year.png", split_by_gender: bool = False) -> None:
    years = []
    number = []
    identity = []
    for year in range(int(min(categories_df.year)), int(max(categories_df.year)) + 1):
        num_year_child = len(categories_df.query(f'(identity == "child") & (year == {year})'))
        num_year_parent = len(categories_df.query(f'(identity == "parent") & (year == {year})'))
        if split_by_gender:
            years.extend([year] * 4)
            male_num_child = len(categories_df.query(f'((identity == "child") & (year == {year})) & (gender == "male")'))
            female_num_child = len(categories_df.query(f'((identity == "child") & (year == {year})) & (gender == "female")'))
            male_num_parent = len(categories_df.query(f'((identity == "parent") & (year == {year})) & (gender == "male")'))
            female_num_parent = len(categories_df.query(f'((identity == "parent") & (year == {year})) & (gender == "female")'))

            identity.append("male child")
            number.append(male_num_child)
            identity.append("female child")
            number.append(female_num_child)

            identity.append("male parent")
            number.append(male_num_parent)
            identity.append("female parent")
            number.append(female_num_parent)

        else:
            years.append(year)
            number.append(num_year_child)
            identity.append("child")

        
            years.append(year)
            number.append(num_year_parent)
            identity.append("parent")

    number_dict = {"year": years, "number": number, "identity": identity}
    number_df = pd.DataFrame(number_dict)
    sns.barplot(data=number_df, x="year", y="number", hue="identity")
    plt.savefig(out_filename)

def words_to_categories_stories(stories_df: pd.DataFrame, out_filename: str) -> None:
    colors = {"harm": "red", "fair": "orange", "loyal": "green", "authority": "blue", "pure": "purple"}

    harm = stories_df.query(f'category == "harm"')
    fair = stories_df.query(f'category == "fairness"')
    loyal = stories_df.query(f'category == "loyalty"')
    authority = stories_df.query(f'category == "authority"')
    pure = stories_df.query(f'category == "purity"')

    values_dict = {"harm": harm, "fair": fair, "loyal": loyal, "authority": authority, "pure": pure}
    quantiles = {i * 0.1: stories_df["flesch_kincaid"].quantile(i * 0.1) for i in range(1, 11)}
    values = {"harm": [], "fair": [], "loyal": [], "authority": [], "pure": []}

    quantiles_lst = []
    moral_types = []
    moral_percentages = []

    for i, (quantile, readability_ind) in enumerate(list(quantiles.items())[:-1]):
        words_included = stories_df.loc[(stories_df["flesch_kincaid"] > readability_ind) & (stories_df["flesch_kincaid"] <= quantiles[(i + 2) * 0.1])]
        #pdb.set_trace()
        for value in values: 
            quantiles_lst.append(quantile * 100)
            value_df = values_dict[value]
            df_year = value_df.loc[(value_df["flesch_kincaid"] > readability_ind) & (value_df["flesch_kincaid"] <= quantiles[(i + 2) * 0.1])]
            
            percentage_value = len(df_year)/len(words_included)
            moral_percentages.append(percentage_value)

            moral_types.append(value)

    cols = {"flesch_kincaid": quantiles_lst, "percentage": moral_percentages, "moral_type": moral_types}
    percentage_df = pd.DataFrame(cols)
    sns.lineplot(data=percentage_df, x="flesch_kincaid", y="percentage", hue="moral_type")
    plt.xlabel("Flesch-Kincaid readability percentile")
    plt.ylabel("Frequency")
    plt.savefig(out_filename)

if __name__ == "__main__":
    categories_v2, cat_names = read_mfd("./data/mfd2.0.dic", version=2, return_categories=True)
    wildcard_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] == "*"}
    pure_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] != "*"}

    childes_df = pickle.load(open("./data/moral_df_2.p", "rb"))

    freq_childes = count_frequencies_childes(childes_df, categories_v2, "./data/mfd-2-childes-freq.csv")

    top_5_words = get_top_k_words_per_value(freq_childes, categories_v2, cat_names, 5, return_list=True)
    top_10_words = get_top_k_words_per_value(freq_childes, categories_v2, cat_names, 10, return_list=True)

    categories_df = pickle.load(open("./data/moral_df_2.p", "rb"))
    story_df = pickle.load(open("./data/moral_df_grimm_v2.p", "rb"))
    #words_to_categories_stories(story_df, ".png")
    #num_words_by_age(categories_df, split_by_gender=True, out_filename="words-per-year-gender.png")

    words_to_categories(categories_df, split_type="orig", person="parent", exclude_hall=True, words_exclude=[], use_frequency_weighting=True)
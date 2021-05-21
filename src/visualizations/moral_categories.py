import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import pickle

from src.data.childes_counts import moral_categories_over_time

def words_to_categories(categories_df: pd.DataFrame) -> None:
    harm = categories_df.query('(type == "harm") & (identity == "parent")')
    fair = categories_df.query('(type == "fairness") & (identity == "parent")')
    loyal = categories_df.query('(type == "loyalty") & (identity == "parent")')
    authority = categories_df.query('(type == "authority") & (identity == "parent")')
    pure = categories_df.query('(type == "purity") & (identity == "parent")')

    years = []
    percentage = []
    moral_type = []
    for year in range(int(min(categories_df.year)), int(max(categories_df.year)) + 1):
        words_year = categories_df.loc[(categories_df["year"] > year) & (categories_df["year"] <= year + 1)]
        if len(words_year) == 0:
            continue
        years.extend([year] * 5)

        percentage_harm = len(harm.loc[(harm["year"] > year) & (harm["year"] <= year + 1)])/len(words_year)
        percentage_fair = len(fair.loc[(fair["year"] > year) & (fair["year"] <= year + 1)])/len(words_year)
        percentage_loyal = len(loyal.loc[(loyal["year"] > year) & (loyal["year"] <= year + 1)])/len(words_year)
        percentage_authority = len(authority.loc[(authority["year"] > year) & (authority["year"] <= year + 1)])/len(words_year)
        percentage_pure = len(pure.loc[(pure["year"] > year) & (pure["year"] <= year + 1)])/len(words_year)

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

    cols_over_time = {"year": years, "percentage": percentage, "moral_type": moral_type}
    data_over_time = pd.DataFrame(cols_over_time)

    sns.lineplot(data=data_over_time, x="year", y="percentage", hue="moral_type")
    plt.savefig("parent-lemma.png")

if __name__ == "__main__":
    categories_df = pickle.load(open("./data/moral_df_lemma.p", "rb"))

    words_to_categories(categories_df)
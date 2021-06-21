import csv
import os
import string
import pdb

from nltk.stem import WordNetLemmatizer
import pandas as pd

from src.data.mfd_read import read_mfd

stories_dir = "data/Grimms_ Fairy Tales/Books/Books"

def moral_categories_over_time(reading_level_df: pd.DataFrame, texts_dict: dict, pure_categories: dict, wildcard_categories: dict) -> pd.DataFrame:
    flesch = []
    g_f = []
    story_ids = []
    words = []
    categories = []
    sentiment = []

    for text in texts_dict:
        story_id = int(text)
        words_list = texts_dict[text].split(" ")
        for word in words_list:
            category = None
            if word in pure_categories:
                category = pure_categories[word]
            elif any(key[:-1] in word for key in wildcard_categories):
                for key in wildcard_categories:
                    if key[:-1] in word:
                        category = wildcard_categories[key]
            if category == None:
                try:
                    flesch.append((reading_level_df.loc[reading_level_df["id"] == story_id])["flesch_kincaid"].item())  
                except:
                    pdb.set_trace()         
                g_f.append((reading_level_df.loc[reading_level_df["id"] == story_id])["gunning_fog"].item()) 
                words.append(word)   
                story_ids.append(story_id)          
                categories.append(None)
                sentiment.append(None)
            else:
                for cat in category:
                    flesch.append((reading_level_df.loc[reading_level_df["id"] == story_id])["flesch_kincaid"].item())           
                    g_f.append((reading_level_df.loc[reading_level_df["id"] == story_id])["gunning_fog"].item()) 
                    words.append(word) 
                    story_ids.append(story_id)       

                    if "virtue" in cat.lower():
                        sentiment.append("pos")
                    elif "vice" in cat.lower():
                        sentiment.append("neg")
                    else:
                        sentiment.append("neu")
                    
                    if "Harm" in cat or "care" in cat:
                        categories.append("harm")
                    elif "Fairness" in cat or "fairness" in cat:
                        categories.append("fairness")
                    elif "Ingroup" in cat or "loyalty" in cat:
                        categories.append("loyalty")
                    elif "Authority" in cat or "authority" in cat:
                        categories.append("authority")
                    elif "Purity" in cat or "sanctity" in cat:
                        categories.append("purity")
                    else:
                        categories.append("other")

    cols = {"story_id": story_ids, "word": words, "flesch_kincaid": flesch, "gunning_fog": g_f, "category": categories, "sentiment": sentiment}

    return pd.DataFrame(cols)

def extract_text(text_dir: str) -> dict:
    lemmatizer = WordNetLemmatizer()
    texts_dict = {}

    for filename in os.listdir(text_dir):
        if not filename.endswith(".txt"):
            continue
        story_id = filename[:-4]
        story_path = os.path.join(stories_dir, filename)
        with open(story_path, "r") as story_file:
            reader = csv.reader(story_file)
            lines = []
            for line in reader:
                if len(line) == 0:
                    continue
                line_removed = "".join([ch for ch in line[0] if (ch != "\n" and ch != "\\") and (ch not in string.punctuation and ch != "\t")])
                words = line_removed.split(" ")
                lemma_words = []
                for word in words:
                    lemmatized = lemmatizer.lemmatize(word)
                    lemma_words.append(lemmatized)

                lines.append(" ".join(lemma_words))

            text = " ".join(lines)
            texts_dict[story_id] = text
    
    return texts_dict

if __name__ == "__main__":
    texts_dict = extract_text(stories_dir)
    reading_level_df = pd.read_csv("./data/stories-grimm.csv")

    categories = read_mfd("./data/mfd.dic")
    wildcard_categories = {cat: val for cat, val in categories.items() if cat[-1] == "*"}
    pure_categories = {cat: val for cat, val in categories.items() if cat[-1] != "*"}

    categories_v2 = read_mfd("./data/mfd2.0.dic", version=2)
    wildcard_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] == "*"}
    pure_categories_v2 = {cat: val for cat, val in categories_v2.items() if cat[-1] != "*"}

    moral_df = moral_categories_over_time(reading_level_df, texts_dict, pure_categories_v2, wildcard_categories_v2)
    moral_df.to_pickle("./data/moral_df_grimm_v2.p")

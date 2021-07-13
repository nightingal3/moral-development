import textstat
import csv
import os
import pdb

import pandas as pd

stories_dir= "data/Grimms_ Fairy Tales/Books/Books"
fb_stories_dir  = "data/Facebook Children Books/cb_train_books"


def get_story_reading_level(story_path: str) -> tuple:
    with open(story_path, "r") as story_file:
        reader = csv.reader(story_file)
        lines = []
        for line in reader:
            if len(line) == 0:
                continue
            line_removed = "".join([ch for ch in line[0] if ch != "\n" and ch != "\\"])
            lines.append(line_removed)

        text = " ".join(lines)
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)

        return flesch_kincaid, gunning_fog

def get_story_characteristics_in_dir(stories_dir: str, out_filename: str) -> None:
    story_number = []
    flesch_kincaid = []
    gunning_fog = []
    for filename in os.listdir(stories_dir):
        if not filename.endswith(".txt"):
            continue
        story_path = os.path.join(stories_dir, filename)
        fk, gf = get_story_reading_level(story_path)
        story_number.append(filename[:-4])
        flesch_kincaid.append(fk)
        gunning_fog.append(gf)

    stories_dict = {"id": story_number, "flesch_kincaid": flesch_kincaid, "gunning_fog": gunning_fog}
    stories_df = pd.DataFrame(stories_dict)

    stories_df.to_csv(out_filename, index=False)

if __name__ == "__main__":
    #get_story_characteristics_in_dir(stories_dir, "./data/stories-grimm.csv")
    get_story_characteristics_in_dir(fb_stories_dir, "./data/stories-fb.csv")

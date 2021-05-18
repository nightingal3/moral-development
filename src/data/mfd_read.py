import csv
import pdb

def read_mfd(in_filename: str) -> dict:
    """Reads the LIWC file associated with the moral foundations dictionary and returns a dictionary
    mapping words/patterns to named categories.

    Args:
        in_filename (str): location of mfd.dic

    Returns:
        dict: maps word/pattern -> category
    """
    categories = {}
    word_to_cat = {}

    with open(in_filename, "r") as dc_file:
        reader = csv.reader(dc_file)
        category_end = 0
        for line in reader:
            if len(line) == 0 or line[0] == "\t\t":
                continue
            if line[0] == "%\t\t":
                category_end += 1    
            if category_end == 1 and line[0] != "%\t\t": # add to category dict
                cat_num, cat_name = line[0].split(None)
                categories[cat_num] = cat_name
            elif category_end == 2 and line[0] != "%\t\t": # done category dict, add to word_dict
                pattern_cats = line[0].split(None)
                pattern = pattern_cats[0]
                cats = pattern_cats[1:]
                word_to_cat[pattern] = [categories[cat_num] for cat_num in cats]

    return word_to_cat

if __name__ == "__main__":
    read_mfd("./data/mfd.dic")
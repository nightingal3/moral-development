import csv

def read_mfd(in_filename: str, version: int = 1, return_categories: bool = False) -> dict:
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
            if (version == 1 and line[0] == "%\t\t") or (version == 2 and line[0] == "%"):
                category_end += 1    
            if category_end == 1 and ((version == 1 and line[0] != "%\t\t") or (version == 2 and line[0] != "%")): # add to category dict
                cat_num, cat_name = line[0].split(None)
                categories[cat_num] = cat_name
            elif category_end == 2 and ((version == 1 and line[0] != "%\t\t") or (version == 2 and line[0] != "%")): # done category dict, add to word_dict
                if version == 1:
                    pattern_cats = line[0].split(None)
                elif version == 2:
                    pattern_cats = line[0].split("\t")
                pattern = pattern_cats[0]
                cats = pattern_cats[1:]
                if pattern not in word_to_cat:
                    word_to_cat[pattern] = [categories[cat_num] for cat_num in cats]
                else:
                    word_to_cat[pattern].extend([categories[cat_num] for cat_num in cats])
    
    if return_categories:
        return word_to_cat, categories
    return word_to_cat

def mfd_overlap(mfd_1_dict: dict, mfd_2_dict: dict) -> tuple:
    """Return the overlap between MFD versions 1 and 2.

    Args:
        mfd_1_dict (dict): MFD 1.0
        mfd_2_dict (dict): MFD 2.0

    Returns:
        tuple: (overlap% mfd2, overlap num words, [list of overlapping words])
    """
    wildcard_categories_1 = {cat: val for cat, val in mfd_1_dict.items() if cat[-1] == "*"}
    pure_categories_1 = {cat: val for cat, val in mfd_1_dict.items() if cat[-1] != "*"}

    overlap_pure = set(pure_categories_1.keys()).intersection(set(mfd_2_dict))
    overlap_wildcard = set()

    for wild_cat in wildcard_categories_1:
        cat_1 = wild_cat[:-1]
        for cat in mfd_2_dict:
            if cat.startswith(cat_1) and cat not in overlap_pure:
                overlap_wildcard.add(cat)

    total_overlap = len(overlap_pure) + len(overlap_wildcard)
    percent_overlap = total_overlap / len(mfd_2_dict)

    return percent_overlap, total_overlap, overlap_pure.update(overlap_wildcard)


if __name__ == "__main__":
    mfd = read_mfd("./data/mfd.dic")
    mfd_2 = read_mfd("./data/mfd2.0.dic", version=2)
    percent, total, _ = mfd_overlap(mfd, mfd_2)
    print(percent, total)
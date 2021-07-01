import pandas as pd
import pickle
import pdb
import string
from nltk.parse import CoreNLPParser

def output_to_txt(childes_dict: pd.DataFrame, out_filename: str) -> None:
    parent_dict = childes_dict["parent"]
    child_dict = childes_dict["child"]
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    all_sentences = []

    for corpus in parent_dict:
        for age in parent_dict[corpus]:
            utterances = parent_dict[corpus][age]
            for u in utterances:
                sentence = [item[0] for item in u if item[0] not in string.punctuation and item[0] != "CLITIC"]
                corenlp_tagged = pos_tagger.tag(sentence)
                sentence_joined = " ".join([f"{word}_{pos}" for word, pos in corenlp_tagged])
                all_sentences.append(sentence_joined)

    with open(out_filename, "w") as out_file:
        for i, sent in enumerate(all_sentences):
            if i == len(all_sentences) - 1:
                out_file.write(sent)
            else:
                out_file.write(sent + " ")


if __name__ == "__main__":
    childes_dict = pickle.load(open("./data/childes-dict-utterances.p", "rb"))
    output_to_txt(childes_dict, "parent_tagged.txt")
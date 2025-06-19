import pandas as pd
from skseq.sequences.label_dictionary import LabelDictionary
from skseq.sequences.sequence_list import SequenceList


def read_sequence_list_from_csv(file_path):
    df = pd.read_csv(file_path)

    word_dict = LabelDictionary()
    tag_dict = LabelDictionary()
    sequence_list = SequenceList(word_dict, tag_dict)

    for _, group in df.groupby("sentence_id"):
        words = group["words"].tolist()
        tags = group["tags"].tolist()
        sequence_list.add_sequence(words, tags, word_dict, tag_dict)

    return sequence_list


def evaluate_corpus(sequences, sequences_predictions):
    """Evaluate classification accuracy at corpus level, comparing with
    gold standard."""
    total = 0.0
    correct = 0.0
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            if sequence.y[j] == y_hat:
                correct += 1
            total += 1
    return correct / total


def evaluate_corpus_non_O(sequences, sequences_predictions):
    total = 0.0
    correct = 0.0
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            gold_label = sequence.y[j]
            if gold_label == 0:
                continue  # skip "O" labels
            if gold_label == y_hat:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def evaluate_O_accuracy(sequences, sequences_predictions):
    total_O = 0
    correct_O = 0
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            if sequence.y[j] == 0:
                total_O += 1
                if y_hat == 0:
                    correct_O += 1
    return correct_O / total_O if total_O > 0 else 0.0


import pandas as pd
from skseq.sequences.label_dictionary import LabelDictionary
from skseq.sequences.sequence_list import SequenceList
from skseq.readers.pos_corpus import PostagCorpus

import codecs
import gzip
from skseq.sequences.label_dictionary import *
from skseq.sequences.sequence import *
from skseq.sequences.sequence_list import *
from os.path import dirname
import numpy as np

"""
New simple version of PostgradCorpus with only needed method to read csv. Any other trial to 
inherit and extend the class failed for me, due to imports (such as os), private methods,
issues in the original load/save methods which don't define what self.int_to_word and
self.int_to_tag are, or other issues...
"""
import os
import codecs
import pandas as pd


class NewPostagCorpus(object):
    """
    Reads a Dataset and saves as attributes of the instantiated corpus

    Attributes:
    -----------
    word_dict: LabelDictionary
        A dictionary with the words in the data

    tag_dict: LabelDictionary
        A dictionary containing all tags (states) in the observed sequences
    """

    def __init__(self):
        # Word dictionary.
        self.word_dict = LabelDictionary()

        # POS tag dictionary.
        self.tag_dict = LabelDictionary()

    def read_sequence_list_csv(self, csv_file, max_sent_len=100, update_dict=True):
        """
        Reads a CSV file with 'sentence_id', 'words', 'tags' columns and returns a SequenceList.
        """
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=["sentence_id", "words", "tags"])

        grouped = df.groupby("sentence_id")

        seq_list = SequenceList(self.word_dict, self.tag_dict)

        for _, group in grouped:
            if len(group) <= 1 or len(group) > max_sent_len:
                continue

            sent_x = []
            sent_y = []
            for _, row in group.iterrows():
                word = row["words"]
                tag = row["tags"]

                if update_dict:
                    if word not in self.word_dict:
                        self.word_dict.add(word)
                    if tag not in self.tag_dict:
                        self.tag_dict.add(tag)
                elif word not in self.word_dict or tag not in self.tag_dict:
                    continue  # Skip unknown tokens in test mode

                sent_x.append(word)
                sent_y.append(tag)
            seq_list.add_sequence(sent_x, sent_y, self.word_dict, self.tag_dict)

        return seq_list


from skseq.sequences.id_feature import IDFeatures


class NERFeatures(IDFeatures):
    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word string
        if isinstance(x, str):
            word = x
        else:
            try:
                word = self.dataset.x_dict.get_label_name(x)
            except IndexError:
                word = "<UNK>"

        feat_id = self.add_feature(f"id:{word}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # hand-crafted for NER:

        # lowercased
        feat_id = self.add_feature(f"lower:{word.lower()}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # Capitalization
        if word and word[0].isupper():
            feat_id = self.add_feature(f"capitalized::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # All caps
        if word.isupper():
            feat_id = self.add_feature(f"all_caps::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Contains digit
        if any(char.isdigit() for char in word):
            feat_id = self.add_feature(f"contains_digit::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Prefixes and suffixes
        for l in [2, 3, 4]:
            if len(word) > l:
                suffix = word[-l:]
                prefix = word[:l]
                feat_id = self.add_feature(f"suffix:{suffix}::{y_name}")
                if feat_id != -1:
                    features.append(feat_id)
                feat_id = self.add_feature(f"prefix:{prefix}::{y_name}")
                if feat_id != -1:
                    features.append(feat_id)

        # === FALLBACK FOR UNKNOWN ===
        if len(features) == 0:
            feat_id = self.add_feature(f"unknown_word::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        return features


def flatten_labels(sequences, tag_dict):
    """Flatten list of Sequence objects to list of label names."""
    labels = []
    for seq in sequences:
        for y in seq.y:
            labels.append(tag_dict.get_label_name(y))
    return labels


def flatten_predictions(sequences_predictions, tag_dict):
    """Flatten predicted sequences to list of label names."""
    labels = []
    for pred_seq in sequences_predictions:
        for y_hat in pred_seq.y:
            labels.append(tag_dict.get_label_name(y_hat))
    return labels

from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_predictions(true_seqs, pred_seqs, tag_dict, set_name="Set"):
    true_labels = flatten_labels(true_seqs, tag_dict)
    pred_labels = flatten_predictions(pred_seqs, tag_dict)

    # Compute confusion matrix
    labels = list(tag_dict.keys())
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    # Plot confusion matrix with seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"Confusion Matrix ({set_name})")
    plt.show()

    # Compute weighted F1-score
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    print(f"{set_name} Weighted F1-score: {f1:.4f}")

    # Optional: detailed classification report
    print(classification_report(true_labels, pred_labels, labels=labels))

    return cm, f1

from skseq.sequences.id_feature import IDFeatures

class NERFeatures_2(IDFeatures):
    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word string
        if isinstance(x, str):
            word = x
        else:
            try:
                word = self.dataset.x_dict.get_label_name(x)
            except IndexError:
                word = "<UNK>"

        # Basic word identity
        feat_id = self.add_feature(f"id:{word}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # Lowercased
        feat_id = self.add_feature(f"lower:{word.lower()}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # Capitalization
        if word and word[0].isupper():
            feat_id = self.add_feature(f"capitalized::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # All caps
        if word.isupper():
            feat_id = self.add_feature(f"all_caps::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Contains digit
        if any(char.isdigit() for char in word):
            feat_id = self.add_feature(f"contains_digit::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Prefixes and suffixes
        for l in [2, 3, 4]:
            if len(word) > l:
                suffix = word[-l:]
                prefix = word[:l]
                feat_id = self.add_feature(f"suffix:{suffix}::{y_name}")
                if feat_id != -1:
                    features.append(feat_id)
                feat_id = self.add_feature(f"prefix:{prefix}::{y_name}")
                if feat_id != -1:
                    features.append(feat_id)

        # Common NER trigger words
        if word in {"Mr", "Mrs", "Dr", "Inc", "Ltd", "Corp"}:
            feat_id = self.add_feature(f"trigger_word::{word}::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Word shape (e.g., Xxxx, xxxx, dddd)
        shape = ''.join(
            ['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else '-' for c in word]
        )
        feat_id = self.add_feature(f"shape:{shape}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # Is a single character
        if len(word) == 1:
            feat_id = self.add_feature(f"single_char::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        # Sentence position
        if pos == 0:
            feat_id = self.add_feature(f"position_start::{y_name}")
        elif pos == len(sequence.x) - 1:
            feat_id = self.add_feature(f"position_end::{y_name}")
        else:
            feat_id = self.add_feature(f"position_middle::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # Previous and next words
        for offset, label in [(-1, "prev"), (1, "next")]:
            if 0 <= pos + offset < len(sequence.x):
                neighbor = sequence.x[pos + offset]
                try:
                    neighbor_word = neighbor if isinstance(neighbor, str) else self.dataset.x_dict.get_label_name(neighbor)
                except IndexError:
                    neighbor_word = "<UNK>"
                feat_id = self.add_feature(f"{label}_word:{neighbor_word.lower()}::{y_name}")
                if feat_id != -1:
                    features.append(feat_id)

        # Fallback for unknown words
        if len(features) == 0:
            feat_id = self.add_feature(f"unknown_word::{y_name}")
            if feat_id != -1:
                features.append(feat_id)

        return features

def show_feats(feature_mapper, seq, inv_feature_dict, feature_type):
    for feat,feat_ids in enumerate(feature_mapper.get_sequence_features(seq)):
        print(feature_type[feat])
        for id_list in feat_ids:
            print ("\t",id_list)
            for k,id_val in enumerate(id_list):
                print ("\t\t", inv_feature_dict[id_val] )
        print("\n")
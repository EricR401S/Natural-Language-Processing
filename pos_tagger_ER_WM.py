"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Eric Rios Soderman & Wafiakmal Miftah, 2022
"""

import nltk
import numpy as np
from typing import Sequence, Tuple, TypeVar

# Importing Training Corpus (The Big Corpus)
training_corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
training_corpus_lower = [
    [(word.lower(), tag) for word, tag in sent] for sent in training_corpus
]
trainlist = [
    ("UNK", "VERB"),
    ("UNK", "DET"),
    ("UNK", "CONJ"),
    ("UNK", "NOUN"),
    ("UNK", "ADV"),
    ("UNK", "NUM"),
    ("UNK", "X"),
    ("UNK", "."),
    ("UNK", "PRT"),
    ("UNK", "PRON"),
    ("UNK", "ADJ"),
    ("UNK", "ADP"),
]
for i in training_corpus_lower:
    for j in i:
        trainlist.append(j)

# Grabbing words
corplist_words = [(tag[0]) for tag in trainlist]

# Grabbing tag
corplist_tag = [tag[1] for tag in trainlist]

# Checking unique words in corpus
corplist_words_set = list(set(corplist_words))

# Checking unique tag in corpus
corplist_tag_set = list(set(corplist_tag))

# Importing Test Sentences
test_corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
test_corpus_lower = [
    [(word.lower(), tag) for word, tag in sent] for sent in test_corpus
]
testlist = []
for a in test_corpus_lower:
    for b in a:
        testlist.append(b)
        pass
    pass
testlist_words = [tag[0] for tag in testlist]

# Editing Input from string to number | Based on unique list of words
numbered_input = []
for word in testlist_words:
    if word not in corplist_words_set:
        numbered_input.append((corplist_words_set.index("UNK")))
        pass
    else:
        numbered_input.append((corplist_words_set.index(word)))
        pass
    pass

# Create ones matrix from the length of the unique tags and the length of the unique words
ones_matrix = np.ones((len(corplist_tag_set), len(corplist_words_set)))

corplist_words_dict = {}
for word in trainlist:
    for item in word:
        if word in corplist_words_dict:
            corplist_words_dict[word] += 1
        else:
            corplist_words_dict[word] = 1
            pass
        pass
    pass

for i in corplist_words_set:
    for j in corplist_tag_set:
        x = (i, j)
        if x in corplist_words_dict:
            a = corplist_words_set.index(i)
            b = corplist_tag_set.index(j)
            ones_matrix[b][a] += corplist_words_dict[x]
            pass
        else:
            pass
    pass

ones_matrix = ones_matrix / np.sum(ones_matrix, axis=0)

# Function to help transition matrix
def t2_given_t1(t2, t1, corpus=trainlist):
    tags = []
    for i in corpus:
        tags.append(i[1])
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags) - 1):
        if tags[index] == t1 and tags[index + 1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# Making pos matrix
tags_matrix = np.ones((len(corplist_tag_set), len(corplist_tag_set)), dtype="float32")
for i, t1 in enumerate(corplist_tag_set):
    for j, t2 in enumerate(corplist_tag_set):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0] / t2_given_t1(t2, t1)[1]
        pass
    pass

corplist_tags_dict = {}
for tag in trainlist:
    if tag[1] in corplist_tags_dict:
        corplist_tags_dict[tag[1]] += 1
        pass
    else:
        corplist_tags_dict[tag[1]] = 1
        pass

pi_matrix = np.ones(len(corplist_tag_set))
for i in corplist_tag_set:
    if i in corplist_tags_dict:
        c = corplist_tag_set.index(i)
        pi_matrix[c] += corplist_tags_dict[i]
        pass
    pass
# pi_matrix = pi_matrix / np.sum(pi_matrix)

# Professor's Provided Viterbi Implementation

Q = TypeVar("Q")
V = TypeVar("V")


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[V, Q], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    # log_d = [np.log(pi) + np.log(B[0, :])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)


# our test

op = viterbi(numbered_input, pi_matrix, tags_matrix, ones_matrix)
sent1 = [[], [], []]
j = 0
for i in op[0]:
    if corplist_tag_set[i] != ".":
        sent1[j].append(corplist_tag_set[i])
    else:
        sent1[j].append(corplist_tag_set[i])
        j += 1

for i in sent1:
    print(f"Our result for the test is {i}")

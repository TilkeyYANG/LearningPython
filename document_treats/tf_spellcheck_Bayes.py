# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:16:53 2018

@author: TilkeyYANG
"""


# Term frequncy
import re, collections

# Cleansing and Lowercase 
# Delete special char
def words(text): return re.findall('[a-z]+', text.lower())

# Problem:
# from (argmaxc P(c|w)) to (argmaxc P(w|c) * P(c) / P(w))
def train(features):
  model = collections.defaultdict(lambda:1)
  for f in features:
    # For not having a prior probability pi=0, so we plus 1 as default
    model[f] += 1
  return model 

# Input big.txt as prior probability
NWORDS = train(words(open('./bigword/big.txt').read()))

# Admissible char list
alphabet = 'abcdefghijklmnopqrstuvwxyz'


# =============================================================================
# Define "Error Distance"
# =============================================================================

# return all the words having ErrorDist = 1
def edits1(word):

  # Check input length
  n = len(word)
  
  # deletion - the: th he te
  set1 = [word[0:i] + word[i+1:] for i in range(n)]
  # transposition - the: hte teh
  set2 = [word[0:i] + word[i+1:] + word[i] + word[i+2:] for i in range(n-1)]
  # alteration - the: tha
  set3 = [word[0:i] + c + word[i+1:] for i in range(n) for c in alphabet]
  # insertion - the: thie
  set4 = [word[0:i] + c + word[i:] for i in range(n+1) for c in alphabet]
  
  return set(set1+set2+set3+set4)
             

# return all the words having ErrorDist = 2
def known_edits2(word):
  return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

 
def known(words): 
  return set(w for w in words if w in NWORDS)


def correct(word):
  candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
  return max(candidates, key=lambda w:NWORDS[w])



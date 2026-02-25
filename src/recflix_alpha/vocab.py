"""
Module: recflix_alpha.vocab
Purpose: Simple token vocabulary and text preprocessing utilities.

`Vocabulary` provides minimal functionality to tokenize text, build a
frequency-thresholded token-to-index mapping, encode text to index
sequences, and save/load the vocabulary dict to disk. It is intentionally
lightweight for quick experiments.
"""

import re
from collections import Counter
import torch


class Vocabulary:
	PAD_TOKEN = "<PAD>"
	UNK_TOKEN = "<UNK>"
	
	def __init__(self,min_freq=2):
		self.min_freq = min_freq
		self.token2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1,}
		self.idx2token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN,}
	
	def tokenize(self,text):
		text = text.lower()
		text = re.sub(r"[^a-z0-9 ]"," ",text)
		tokens = text.split()
		return tokens
	
	def build_vocab(self,texts):
		counter = Counter()
		
		for text in texts:
			tokens = self.tokenize(text)
			counter.update(tokens)
			idx = 2
		
		for token, freq in counter.items():
			if freq >= self.min_freq:
				self.token2idx[token] = idx
				self.idx2token[idx] = token
				idx+=1
		print(f"Vocabulary size : {len(self.token2idx)}")
		
	def encode(self,text):
		tokens = self.tokenize(text)
		return [self.token2idx.get(token,self.token2idx[self.UNK_TOKEN]) for token in tokens]
		
	def save(self,path):
		torch.save(self.token2idx,path)
	def load(self,path):
		self.token2idx = torch.load(path)
		self.idx2token = {idx: token for token, idx in self.token2idx.items()}
		
	

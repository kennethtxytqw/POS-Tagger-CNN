# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>
# Referred to https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import os
import math
import sys
import re
import datetime
from random import sample

from Pos_Tagger import Sentence, Tagged_Sentence, Indexer, Pos_Tagger, to_char_indexes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def errprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class POS_Tagged_Corpus(torch.utils.data.dataset.Dataset):    
    def __init__(self, file):
        super(POS_Tagged_Corpus, self).__init__()
        self._tagged_sentences = []
        self._vocab = set()
        self._tags = set()
        self._chars = set()

        self.len_longest_sent = 0
        self.len_longest_word = 0
        
        self._tags.add(" ")
        self._tags.add(" ")
        self._chars.add(" ")
        self._vocab.add(" ")
        self.consume_file(file)

        self.char_indexer = Indexer(self.chars())
        self.tag_indexer = Indexer(self.tags())
        self.word_indexer = Indexer(self.vocab())

        self.sents = []
        self.sentence_tags = []
        self.data = []
        for tagged_sentence in self.get_tagged_sentences():
            words = tagged_sentence.words()
            tags = tagged_sentence.tags()
            for i in range(self.len_longest_sent - len(tags)):
                tags.append(" ")
                words.append(" ")

            tag_indexes =  torch.tensor(self.tag_indexer.prepare_sequence(tags), dtype=torch.long).to(device)
            self.data.append((words, tag_indexes))

        errprint(self)

    def consume_file(self, file):
        with open(file, "r") as f:
            for line in f:
                if len(line) != 0:
                    self.consume_line(line)

    def consume_line(self, line):
        words = []
        tags = []

        arr = line.rstrip().split(" ")
        
        for word_and_pos in arr:
            word, tag = word_and_pos.rsplit("/", 1)
            self._vocab.add(word)
            self._tags.add(tag)

            words.append(word)
            tags.append(tag)

            for c in word:
                self._chars.add(c)

            self.len_longest_word = len(word) if len(word) > self.len_longest_word else self.len_longest_word
    
        self.len_longest_sent = len(tags) if len(tags) > self.len_longest_sent else self.len_longest_sent

        self._tagged_sentences.append(Tagged_Sentence(words=words, tags=tags))

    def get_tagged_sentences(self, randomly_reordered = False):
        if randomly_reordered:
            return sample(self._tagged_sentences, len(self._tagged_sentences))
        else:
            return self._tagged_sentences

    def vocab(self):
        return self._vocab

    def tags(self):
        return self._tags

    def chars(self):
        return self._chars

    def __str__(self):
        return "POS_Tagged_Corpus object: " + str(len(self.get_tagged_sentences())) + " sentences, " + str(len(self.vocab())) + " words, " + str(len(self.tags())) + " tags, " + str(len(self.chars())) + " chars" + ", len_longest_word=" + str(self.len_longest_word) + ", len_longest_sentence=" + str(self.len_longest_sent)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def train(pos_tagger, pos_tagged_corpus):
    lr = pos_tagger.lr
    total_epoch = pos_tagger.total_epoch

    loss_function = nn.NLLLoss(reduction = "elementwise_mean").to(device)
    optimizer = optim.SGD(pos_tagger.parameters(), lr=lr)

    data_loader = DataLoader(pos_tagged_corpus, batch_size=pos_tagger.batch_size, shuffle=True)

    total = len(data_loader) * total_epoch
    count = 0
    start_time = datetime.datetime.now()
    for epoch in range(total_epoch):
        for i, data in enumerate(data_loader):
            sents, sentence_tagss = data
            sents = [list(sent) for sent in zip(*sents)]
            # Should implement a randomizer that replaces word with <UNK> to increase some depedency on char level representation
            # If time permits
            sents_in_word_indexes = torch.tensor([pos_tagged_corpus.word_indexer.prepare_sequence(sent) for sent in sents]).to(device)
            sents_in_char_indexes = to_char_indexes(sents, pos_tagged_corpus.char_indexer, pos_tagged_corpus.len_longest_word).to(device)

            pos_tagger.zero_grad()
            pos_tagger.reinit_hidden(sents_in_word_indexes.size()[0])
            
            tag_scores = pos_tagger(sents_in_word_indexes, sents_in_char_indexes)

            # transposition_start = datetime.datetime.now()
            transposed_tag_scores = tag_scores.transpose(1,2)
            # errprint("Transposition time wasted: ", datetime.datetime.now() - transposition_start)

            # loss_start = datetime.datetime.now()
            loss = loss_function(transposed_tag_scores, sentence_tagss)
            loss.backward()
            # errprint("Loss time wasted: ", datetime.datetime.now() - loss_start)

            optimizer.step()

            count += 1
            end_time = datetime.datetime.now()
            errprint('Estimated time left:', (end_time - start_time)/ count * (total - count))
            errprint("Loss:", loss)
            errprint("Training:", count/total*100, "%")

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
		# use torch library to save model parameters, hyperparameters, etc. to model_file

    start_time = datetime.datetime.now()

    pos_tagged_corpus = POS_Tagged_Corpus(train_file)
    end_time = datetime.datetime.now()
    print('Data Preparation Time:', end_time - start_time)

    pos_tagger = Pos_Tagger(word_indexer=pos_tagged_corpus.word_indexer,
                            tagset_size = pos_tagged_corpus.tag_indexer.size(), 
                            len_longest_sent=pos_tagged_corpus.len_longest_sent,
                            char_indexer=pos_tagged_corpus.char_indexer, 
                            len_longest_word=pos_tagged_corpus.len_longest_word,
                            device=device,
                            pad=pos_tagged_corpus.char_indexer.index(" "))

    pos_tagger.to(device)

    # Start training and set these Hyperparameters
    train(pos_tagger, pos_tagged_corpus)

    # Save model and indices
    torch.save({
        'pos_tagger': pos_tagger.state_dict(),
        'word2index': pos_tagged_corpus.word_indexer.get_dict(),
        'tag2index': pos_tagged_corpus.tag_indexer.get_dict(),
        'char2index': pos_tagged_corpus.char_indexer.get_dict(),
        'len_longest_sent': pos_tagged_corpus.len_longest_sent,
        'len_longest_word': pos_tagged_corpus.len_longest_word
    }, model_file)

    print('Finished...')
    
    end_time = datetime.datetime.now()
    print('Build Tagger Time:', end_time - start_time)
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)


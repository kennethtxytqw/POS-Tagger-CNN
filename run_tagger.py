# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch
import datetime
from torch.utils.data import Dataset, DataLoader

from Pos_Tagger import Pos_Tagger, Indexer, Tagged_Sentence, Sentence, to_char_indexes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestSet(torch.utils.data.dataset.Dataset):    
    def __init__(self, file, word_indexer, len_longest_sent):
        super(TestSet, self).__init__()
        self.data = []
        
        self.word_indexer = word_indexer
        self.len_longest_sent = len_longest_sent
        self.consume_file(file)

    def consume_file(self, file):
        with open(file, "r") as f:
            for line in f:
                if len(line) != 0:
                    self.consume_line(line)

    def consume_line(self, line):
        words = line.rstrip().split(" ")
        self.data.append(words)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file

    
    start_time = datetime.datetime.now()
    checkpoint = torch.load(model_file)
    word_indexer = Indexer(d=checkpoint['word2index'])
    char_indexer = Indexer(d=checkpoint['char2index'])
    tag_indexer = Indexer(d=checkpoint['tag2index'])
    len_longest_sent = checkpoint['len_longest_sent']
    len_longest_word = checkpoint['len_longest_word']

    pos_tagger = Pos_Tagger(word_indexer=word_indexer,
                            tagset_size = tag_indexer.size(), 
                            len_longest_sent=len_longest_sent,
                            char_indexer=char_indexer, 
                            len_longest_word=len_longest_word,
                            device=device,
                            pad=char_indexer.index(" "))

    pos_tagger.load_state_dict(checkpoint['pos_tagger'])
    pos_tagger = pos_tagger.to(device)
    
    tagged_sentences = []
    data_loader = DataLoader(TestSet(test_file, word_indexer, len_longest_sent), batch_size=1)
    for i, sents in enumerate(data_loader):
        sents = [list(sent) for sent in zip(*sents)]
        sents_in_word_indexes = torch.tensor([word_indexer.prepare_sequence(sent) for sent in sents]).to(device)
        sents_in_char_indexes = to_char_indexes(sents, char_indexer, len_longest_word).to(device)
        pos_tagger.reinit_hidden(len(sents))
        
        tag_scores = pos_tagger(sents_in_word_indexes, sents_in_char_indexes)
        # print("tag_scores: B x sent_len x tagset_size", tag_scores.size())
        tags = get_tags(tag_scores, tag_indexer)
        for j, sent in enumerate(sents):
            tagged_sentence = Tagged_Sentence(words=sent, tags=tags[j])
            tagged_sentences.append(tagged_sentence)

    with open(out_file, 'w') as of:
        for tagged_sentence in tagged_sentences:
            of.write(tagged_sentence.to_string())
            of.write('\n')
    print('Finished...')
    
    end_time = datetime.datetime.now()
    print('Run Tagger Time:', end_time - start_time)

def get_tags(tag_scores, tag_indexer):
    tags_indices = torch.argmax(tag_scores, dim =2)
    return [tag_indexer.read_seq(batch_tag_indices) for batch_tag_indices in tags_indices]


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)

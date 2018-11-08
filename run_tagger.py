# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch
import datetime


from Pos_Tagger import Pos_Tagger, Indexer, Tagged_Sentence, Sentence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
    pos_tagger.reinit_hidden(1)
    tagged_sentences = []
    with open(test_file, 'r') as tf:
        for line in tf:
            words = line.rstrip().split(" ")
            sent_in_word_indexes = word_indexer.prepare_sequence(words, pad=" ", length=len_longest_sent).to(device)
            tag_scores = pos_tagger(sent_in_word_indexes.view(-1, 1))
            tags = get_tags(tag_scores, tag_indexer)
            tagged_sentence = Tagged_Sentence(words=words, tags=tags)
            tagged_sentences.append(tagged_sentence)

    with open(out_file, 'w') as of:
        for tagged_sentence in tagged_sentences:
            of.write(tagged_sentence.to_string())
            of.write('\n')
    print('Finished...')
    
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)

def get_tags(tag_scores, tag_indexer):
    tags = []
    for tag_scores_of_word in tag_scores:
        tags.append(tag_indexer.element(torch.argmax(tag_scores_of_word)))
    return tags


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)

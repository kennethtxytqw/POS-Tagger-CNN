import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime, sys

def errprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Pos_Tagger(nn.Module):    
    def __init__(self, word_indexer, tagset_size, len_longest_sent, char_indexer=None,  len_longest_word=None, pad=0, device="cpu"):
        super(Pos_Tagger, self).__init__()
        self.benchmark = True
        self.cuda = True
        self.device = device

        # Hyperparameters
        self.batch_size = 8
        self.lr = 0.2
        self.total_epoch = 1
        self.word_dim = 256
        self.hidden_dim = 128
        self.char_dim = 16
        self.window_size = 3
        self.conv_filters_size = 64
        self.with_char_level = False

        self.tagset_size = tagset_size
        self.word_indexer = word_indexer
        self.word_vocab_size = self.word_indexer.size()
        self.len_longest_sent = len_longest_sent

        self.word_embeddings = nn.Embedding(self.word_vocab_size, self.word_dim)
        self.word_embeddings.weight.data[word_indexer.index("<UNK>")] = 0
        if self.with_char_level:
            # With char level representation
            self.len_longest_word = len_longest_word
            self.char_indexer = char_indexer
            self.char_vocab_size = self.char_indexer.size()
            self.char_embeddings = nn.Embedding(self.char_vocab_size, self.char_dim)
            self.lstm= nn.LSTM(input_size = self.word_dim + self.conv_filters_size, hidden_size = self.hidden_dim//2 , bidirectional=True, batch_first=True, bias=True)
            self.conv = nn.Conv1d(1, self.conv_filters_size, self.window_size * self.char_dim, bias=True)
            self.padding_layer = nn.ConstantPad1d((self.window_size-1)//2, pad)
        else:
            # Without char level representation
            self.lstm= nn.LSTM(input_size = self.word_dim, hidden_size = self.hidden_dim//2 , bidirectional=True, batch_first=True, bias=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size, bias=True)

    def reinit_hidden(self, curr_batch_size):
        self.hidden = (torch.zeros(2, curr_batch_size, self.hidden_dim//2).to(self.device),
                torch.zeros(2, curr_batch_size, self.hidden_dim//2).to(self.device))

    def forward_char_level(self, word_in_char_indexes):
        char_embeds = self.char_embeddings(self.padding_layer(word_in_char_indexes))
        # errprint("char_embeds", char_embeds.size())
        conv_result = self.conv(char_embeds.view(char_embeds.size()[0], 1, -1))
        
        # errprint("conv_result", conv_result.size())
        return conv_result.max(2)[0]

    def forward(self, sents_in_w_indices, sents_in_char_indexes=None):
        # Expect sent to be a sequence of word indexes
        curr_batch_size = sents_in_w_indices.size()[0]
        sent_len = sents_in_w_indices.size()[1]
        # errprint("sents_in_w_indices: B x sent_len", sents_in_w_indices.size())
        word_embeds = self.word_embeddings(sents_in_w_indices)
        # errprint("word_embeds: B x sent_len x word_dim", word_embeds.size())
        if self.with_char_level:
            # errprint("sents_in_char_indexes: B x sent_len x word_len", sents_in_char_indexes.size())
            sents_in_char_indexes = sents_in_char_indexes.view(curr_batch_size * sent_len, self.len_longest_word).to(self.device)
            # errprint("sents_in_char_indexes: (B * sent_len) x word_len", sents_in_char_indexes.size())
            sents_in_char_level = self.forward_char_level(sents_in_char_indexes)
            sents_in_char_level = sents_in_char_level.view(curr_batch_size, sent_len, self.conv_filters_size)

            # errprint("sents_in_char_level: B x sent_len x conv_size", sents_in_char_level.size())
            lstm_in = torch.cat((word_embeds, sents_in_char_level), dim = 2).to(self.device)
            # errprint("lstm_in: B x sent_len x (word_dim + conv_size)", lstm_in.size())
        else:
            lstm_in = word_embeds
            # errprint("lstm_in: B x sent_len x word_dim", lstm_in.size())

        # start_time = datetime.datetime.now()
        # errprint("hidden: 2 x B x hidden_dim//2", self.hidden[0].size())
        lstm_out, self.hidden = self.lstm(lstm_in, self.hidden)
        lstm_out = lstm_out.to(self.device)
        # errprint("lstm_out: B x sent_len x (2*hidden_dim)", lstm_out.size())

        # end_time = datetime.datetime.now()
        # errprint('LSTM Time:', end_time - start_time)

        # start_time = datetime.datetime.now()
        tag_space = self.hidden2tag(lstm_out)
        # errprint("tag_space: B x sent_len x tagset_size", tag_space.size())
        tag_scores = F.log_softmax(tag_space, dim=2)
        # errprint("tag_scores: B x sent_len x tagset_size", tag_scores.size())
        # end_time = datetime.datetime.now()
        # errprint('Hidden2tag2logsoftmax Time:', end_time - start_time)

        return tag_scores

class Indexer(object):
    def __init__(self, vocab = None, d = None):
        if d:
            self._element2index = d
            self._vocab = list(d.keys())
            for key, value in d.items():
                self._vocab[value] = key
        else:
            self._element2index = {}
            if vocab != None:
                vocab = list(vocab)
                vocab.append('<UNK>')
                self._vocab = vocab
                for index, element in enumerate(vocab):
                    self._element2index[element] = index

    def element(self, index):
        return self._vocab[index]

    def index(self, element):
        try:
            return self._element2index[element]
        except:
            return self._element2index['<UNK>']

    def prepare_sequence(self, seq, pad=None, length=None):
        if length:
            indexes = np.zeros(length)
            for i in range(len(seq)):
                if i == length:
                    break
                indexes[i] = self.index(seq[i])
            return indexes
        else:
            return [self.index(element) for element in seq]

    def get_dict(self):
        return self._element2index

    def size(self):
        return len(self._element2index)

    def read_seq(self, seq):
        return [self.element(index) for index in seq]

class Sentence(object):
    def __init__(self, string = None, words = None):
        if string != None:
            self.words(string.rstrip().split(" "))
        if words != None:
            self.words(words)

    def words(self, words = None):
        if words != None:
            self._words = words
        return self._words

    def __str__(self):
        return self.words().__str__()

class Tagged_Sentence(Sentence):
    
    def __init__(self, words_string = None, tags_string = None, words = None, tags = None):
        super(Tagged_Sentence, self).__init__(string=words_string, words=words)
        if tags_string != None:
            self.tags(tags_string.rstrip().split(" "))
        if tags != None:
            self.tags(tags)

    def tags(self, tags = None):
        if tags != None:
            self._tags = tags
        return self._tags

    def __str__(self):
        return self.words().__str__() + "\n" + self.tags().__str__() + "\n"

    def to_string(self):
        return " ".join([ word + '/' + self._tags[i] for i, word in enumerate(self.words())])

def to_char_indexes(sents_in_w, char_indexer, len_longest_word):
    # start_time = datetime.datetime.now()
    sents_in_char_indexes = []
    for sent in sents_in_w:
        sent_in_char_indexes = [char_indexer.prepare_sequence(word, pad=" ", length=len_longest_word) for word in sent]
        sents_in_char_indexes.append(sent_in_char_indexes)
    sents_in_char_indexes = torch.tensor(sents_in_char_indexes, dtype=torch.long)
    # end_time = datetime.datetime.now()

    # errprint('Prepping char indexes Time:', end_time - start_time)
    return sents_in_char_indexes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Pos_Tagger(nn.Module):    
    def __init__(self, word_indexer, tagset_size, len_longest_sent, char_indexer=None,  len_longest_word=None, pad=0, device="cpu"):
        super(Pos_Tagger, self).__init__()
        self.benchmark = True
        self.cuda = True
        self.device = device

        # Hyperparameters
        self.batch_size = 16
        self.lr = 0.2
        self.total_epoch = 5
        self.word_dim = 200
        self.hidden_dim = 100
        self.char_dim = 10
        self.window_size = 3
        self.conv_filters_size = 20
        self.with_char_level = True

        self.tagset_size = tagset_size
        self.word_indexer = word_indexer
        self.word_vocab_size = self.word_indexer.size()
        self.len_longest_sent = len_longest_sent

        self.word_embeddings = nn.Embedding(self.word_vocab_size, self.word_dim)
        if self.with_char_level:
            # With char level representation
            assert char_indexer
            assert len_longest_word
            self.len_longest_word = len_longest_word
            self.char_indexer = char_indexer
            self.char_vocab_size = self.char_indexer.size()
            self.char_embeddings = nn.Embedding(self.char_vocab_size, self.char_dim)
            self.lstm= nn.LSTM(self.word_dim + self.conv_filters_size, self.hidden_dim//2 , bidirectional=True)
            self.conv = nn.Conv1d(1, self.conv_filters_size, self.window_size * self.char_dim)
            self.padding_layer = nn.ConstantPad1d((self.window_size-1)//2, pad)
        else:
            # Without char level representation
            self.lstm= nn.LSTM(self.word_dim, self.hidden_dim//2 , bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def reinit_hidden(self, curr_batch_size):
        self.hidden = (torch.zeros(2, curr_batch_size, self.hidden_dim//2).to(self.device),
                torch.zeros(2, curr_batch_size, self.hidden_dim//2).to(self.device))

    def forward_char_level(self, word_in_char_indexes):
        char_embeds = self.char_embeddings(self.padding_layer(word_in_char_indexes.to(self.device)))
        # print("char_embeds", char_embeds.size())
        conv_result = self.conv(char_embeds.view(char_embeds.size()[0], 1, -1))
        
        # print("conv_result", conv_result.size())
        return conv_result.max(2)[0]

    def forward(self, sent_in_word_indexes):
        # Expect sent to be a sequence of word indexes
        curr_batch_size = sent_in_word_indexes.size()[1]
        sent_in_word_indexes = sent_in_word_indexes.to(self.device)
        # print("sent_in_word_indexes", sent_in_word_indexes.size())
        word_embeds = self.word_embeddings(sent_in_word_indexes)
        # print("word_embeds", word_embeds.size())
        if self.with_char_level:
            print("sent_in_word_indexes", sent_in_word_indexes.size())
            sents_in_char_level = []
            for seq in sent_in_word_indexes.t():
                sent_in_char_indexes = []
                for word_index in seq:
                    word_in_char_index = self.char_indexer.prepare_sequence(self.word_indexer.element(word_index), pad=" ", length=self.len_longest_word)
                    sent_in_char_indexes.append(word_in_char_index)
                sent_in_char_indexes =  torch.stack(sent_in_char_indexes).to(self.device)
                sent_in_char_level = self.forward_char_level(sent_in_char_indexes).to(self.device)
                sents_in_char_level.append(sent_in_char_level)
                
                # print("sent_in_char_indexes", sent_in_char_indexes.size())
                # print("sent_in_char_level", sent_in_char_level.size())
            sents_in_char_level = torch.stack(sents_in_char_level)
            # print("sents_in_char_level", sents_in_char_level.size())
            lstm_in = torch.cat((word_embeds, sents_in_char_level.transpose(0,1)), dim = 2).to(self.device)
        else:
            lstm_in = word_embeds
            
        # print("lstm_in", lstm_in.size())
        lstm_out, self.hidden = self.lstm(lstm_in, self.hidden)
        lstm_out = lstm_out.to(self.device)

        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        
        # print("lstm_in", lstm_in.size())
        # print("lstm_out", lstm_out.size())
        # print("tag_space", tag_space.size())
        # print("tag_scores", tag_scores.size())
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
                vocab.sort()
                self._vocab = vocab
                for index, element in enumerate(vocab):
                    self._element2index[element] = index

    def element(self, index):
        return self._vocab[index]

    def index(self, element):
        if element in self._element2index:
            return self._element2index[element]
        else:
            return self._element2index['<UNK>']

    def prepare_sequence(self, seq, pad=None, length=None):
        indexes = [self.index(element) for element in seq]
        if length:
            if len(indexes) > length:
                indexes = indexes[:length]
            else:
                for i in range (length - len(indexes)):
                    indexes.append(self.index(pad))
        return torch.tensor(indexes, dtype=torch.long)

    def get_dict(self):
        return self._element2index

    def load_dict(self, d):
        self._element2index = d

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
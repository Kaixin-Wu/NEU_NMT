from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
##from nltk.tokenize import word_tokenize

def load_dataset(batch_size):

    ##def tokenize_nltk(text):
    ##    return word_tokenize(text)

    DE = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')
    EN = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')

    train = TranslationDataset(path='data/40w/train', exts=('.ch', '.en'), fields=(DE, EN))
    val = TranslationDataset(path='data/40w/valid', exts=('.ch', '.en'), fields=(DE, EN))
    test = TranslationDataset(path='data/40w/test', exts=('.ch', '.en'), fields=(DE, EN))

    #DE.build_vocab(train.src, min_freq=2)
    DE.build_vocab(train.src, max_size=30000)
    EN.build_vocab(train.trg, max_size=30000)

    train_iter, val_iter, test_iter = BucketIterator.splits((train, val, test),
                                                            batch_size=batch_size, repeat=False)

    return train_iter, val_iter, test_iter, DE, EN

##if __name__ == "__main__":
##    load_dataset(batch_size=32)

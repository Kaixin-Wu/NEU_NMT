from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset

def load_dataset(batch_size):
    '''
    	load data sets.
    '''

    Lang1 = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')
    Lang2 = Field(include_lengths=True, init_token='<sos>', eos_token='<eos>')

    train = TranslationDataset(path='data-3w/train', exts=('.de', '.en'), fields=(Lang1, Lang2))
    val = TranslationDataset(path='data-3w/valid', exts=('.de', '.en'), fields=(Lang1, Lang2))
    test = TranslationDataset(path='data-3w/test', exts=('.de', '.en'), fields=(Lang1, Lang2))

    Lang1.build_vocab(train.src, min_freq=2)
    Lang2.build_vocab(train.trg, max_size=10000)

    train_iter, val_iter, test_iter = BucketIterator.splits((train, val, test),
                                                            batch_size=batch_size, repeat=False)

    return train_iter, val_iter, test_iter, Lang1, Lang2

#coding=utf8
import argparse, os, sys, pickle, json
import abc
import functools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from gensim.models.keyedvectors import KeyedVectors
from resources import vncorenlp

class Embedder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def tokenize(self, sentence):
        '''Given a string, return a list of tokens suitable for lookup.'''
        pass

    @abc.abstractmethod
    def untokenize(self, tokens):
        '''Undo tokenize.'''
        pass

    @abc.abstractmethod
    def contains(self, token):
        pass

    @abc.abstractmethod
    def to(self, device):
        '''Transfer the pretrained embeddings to the given device.'''
        pass

class PhoW2V(Embedder):
    def __init__(self, emb_path):
        load_w2v = KeyedVectors.load_word2vec_format(emb_path, binary=False)
        load_w2v.init_sims(replace=True)
        w2v_path = os.path.join(emb_path[:emb_path.rindex("/")], 'w2v')
        load_w2v.save(w2v_path)
        self.phoemb = KeyedVectors.load(w2v_path, mmap='r')
        self.dim = self.phoemb.vector_size
        self.vocab = set(self.phoemb.key_to_index.keys())
        self.vnnlp = vncorenlp.VNCoreNLP()

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        ann = vncorenlp.tokenize(text)
        return [tok.lower() for sent in ann for tok in sent]

    @functools.lru_cache(maxsize=1024)
    def tokenize_for_copying(self, text):
        ann = self.vnnlp.tokenize(text)
        text = [tok.lower() for sent in ann for tok in sent]
        text_for_copying = [tok.lower() for sent in ann for tok in sent]
        return text, text_for_copying

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def contains(self, token):
        return token in self.vocab

    def to(self, device):
        self.vectors = self.vectors.to(device)

def construct_vocab_from_dataset(*data_paths, table_path='data/tables.bin', mwf=4, reference_file=None, output_path=None, sep='\t'):
    phow2v = PhoW2V(reference_file)

    words = []
    tables = pickle.load(open(table_path, 'rb'))
    for fp in data_paths:
        dataset = pickle.load(open(fp, 'rb'))
        for ex in dataset:
            words.extend(ex['processed_question_toks'])
            db = tables[ex['db_id']]
            words.extend(['table'] * len(db['table_names']))
            words.extend(db['column_types'])
            for c in db['processed_column_toks']:
                words.extend(c)
            for t in db['processed_table_toks']:
                words.extend(t)
    cnt = Counter(words)
    vocab = sorted(list(cnt.items()), key=lambda x: - x[1])
    phow2v_vocab = phow2v.vocab
    oov_words, oov_but_freq_words = set(), []
    for w, c in vocab:
        if w not in phow2v_vocab:
            oov_words.add(w)
            if c >= mwf:
                oov_but_freq_words.append((w, c))
    print('Out of glove vocabulary size: %d\nAmong them, %d words occur equal or more than %d times in training dataset.' % (len(oov_words), len(oov_but_freq_words), mwf))
    with open(output_path, 'w') as of:
        # first serialize oov but frequent words, allowing fine-tune them during training
        for w, c in oov_but_freq_words:
            of.write(w + sep + str(c) + '\n')
        # next serialize words in both train vocab and glove vocab according to decreasing frequency
        for w, c in vocab:
            if w not in oov_words:
                of.write(w + sep + str(c) + '\n')
    return len(vocab)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', nargs='+', type=str, help='input preprocessed dataset file')
    parser.add_argument('--table_path', type=str, default='data/tables.bin', help='preprocessed table file')
    parser.add_argument('--output_path', type=str, required=True, help='output word vocabulary path')
    parser.add_argument('--reference_file', type=str, default='pretrained_models/glove-42b-300d/vocab_glove.txt',
        help='eliminate word not in glove vocabulary, unless it occurs frequently >= mwf')
    parser.add_argument('--mwf', default=4, type=int,
        help='minimum word frequency used to pick up frequent words in training dataset, but not in glove vocabulary')
    args = parser.parse_args()

    construct_vocab_from_dataset(*args.data_paths, table_path=args.table_path, mwf=args.mwf, reference_file=args.reference_file, output_path=args.output_path)
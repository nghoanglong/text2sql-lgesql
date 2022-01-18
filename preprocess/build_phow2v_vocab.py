#coding=utf8
import argparse, os, sys, pickle, json
import abc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from gensim.models.keyedvectors import KeyedVectors

class Embedder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def contains(self, token):
        pass
    
    @abc.abstractmethod
    def lookup(self, token):
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

    def lookup(self, token):
        try:
            return self.phoemb.get_vector(token)
        except:
            return None

    def contains(self, token):
        return token in self.vocab

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
    with open(output_path, 'w', encoding='utf-8') as of:
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
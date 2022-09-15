from dict_index import DictIndex
from matrix_index import MatrixIndex
from preprocessing import *
from file_reader import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Enter data directory')
args = parser.parse_args()

def main():
    corpus = {name: preprocess_file(text) for (name, text) in read_files(args.directory)}

    for ind_cls in [DictIndex, MatrixIndex]:
        index = ind_cls(corpus)
        print('Most frequent word:', index.most_frequent())
        print('Least frequent word:', index.least_frequent())
        print('Universal words:', ' '.join(index.universal_tokens()))
        corr = {'Monika': 'моника мон',
                'Rachel': 'рэйчел рейч',
                'Chandler': 'чендлер чэндлер чен',
                'Phoebe': 'фиби фибс',
                'Ross': 'росс',
                'Joe': 'джоуи джои джо'}
        freqs = {name: index.process_query(query) for (name, query) in corr.items()}
        print('Most popular character:', max(list(freqs.keys()), key= lambda x: freqs[x]))

if __name__ == '__main__':
    main()

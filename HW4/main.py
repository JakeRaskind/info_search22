from matrix_index import MatrixIndex
from preprocessing import preprocess_file
from file_reader import read_file
import numpy as np
import argparse


def main():
    corpus = read_file(args.directory)
    index = MatrixIndex(corpus, args.index_mode)
    queries = args.list
    for query in queries:
        query = ' '.join(query)
        ans_list = index.process_query(query)
        print('Query: ' + query)
        print('The relevant answers are:')
        print('\n'.join(ans_list[:10]), '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Enter data directory')
    parser.add_argument('index_mode', help='''Enter index mode: 'bm' for Okapi-BM25, 'bert' for BERT-based index''')
    parser.add_argument('-l', '--list', action='append', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    main()
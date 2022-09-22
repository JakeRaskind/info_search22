from matrix_index import MatrixIndex
from preprocessing import preprocess_file
from file_reader import read_files
import numpy as np
import argparse


def main():
    corpus = {name: preprocess_file(text) for (name, text) in read_files(args.directory)}
    index = MatrixIndex(corpus)
    queries = args.list
    for query in queries:
        query = ' '.join(query)
        ep_list = index.process_query(query)
        print('Query: ' + query)
        print('The most similar episodes are:')
        print('\n'.join(ep_list[:10]), '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Enter data directory')
    parser.add_argument('-l', '--list', action='append', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    main()
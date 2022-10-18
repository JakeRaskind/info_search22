from matrix_index import MatrixIndex
from preprocessing import preprocess_file
from file_reader import read_file
import numpy as np
import argparse
import os

os.environ['no_proxy'] = '*'
parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Enter indices directory')
args = parser.parse_args()
index = MatrixIndex(args.directory)
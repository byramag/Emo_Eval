import sys
from emo_eval import main, parseArgs

file_name, sample_size = parseArgs(sys.argv)
main("data/train.txt", sample_size=sample_size)

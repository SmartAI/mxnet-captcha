import sys
import os
import glob

with open('flist.txt', 'w') as fin:
    jpgs = glob.glob('{}/*.jpg'.format(sys.argv[1]))
    index = 0
    for jpg in jpgs:
        labels = os.path.splitext(jpg)
        labels = list(os.path.basename(labels[0]))
        texts = '{}\t{}\t{}\n'.format(index, '\t'.join(labels), jpg)
        index +=1
        fin.write(texts)

import csv
import numpy as np

file = "plaintext.txt"


freqs = {}

with open(file, 'rb') as reader:
    chars = [i for i in reader.read() if i != '\n']

for char in chars:
    if char not in freqs:
        freqs[char] = 1

    else:
        freqs[char] += 1


for k, v in freqs.iteritems():
    freqs[k] = v / float(len(chars))

for k, v in freqs.iteritems():
    print "k: {0}, freq: {1}".format(k, v)


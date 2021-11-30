import nltk
import sys
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

print(" Arguments are : ", sys.argv[1], sys.argv[2])
# There can be multiple references but 1 hypothesis
reference_file = open(sys.argv[1], 'r')
reference_lines = reference_file.readlines()

hypothesis_file = open(sys.argv[2], 'r')
hypothesis_lines = hypothesis_file.readlines()
count = 0
references = []
hypothesis = []
pair_reference = []
for line in reference_lines:
    if count%2 != 0:
        references.append(pair_reference)
        pair_reference = []

    pair_reference.append(line.split())
    count += 1
    # references.append([line.split()])

print(references)
for line in hypothesis_lines:
    hypothesis.append(line.split())

print(hypothesis)

cc = SmoothingFunction()

bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypothesis,weights = [1], smoothing_function=cc.method3)
print(bleu_score)


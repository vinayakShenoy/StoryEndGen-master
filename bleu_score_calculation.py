import nltk
import sys
import os
from datetime import datetime
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

print(" Arguments are : ", sys.argv[1], sys.argv[2])
# There can be multiple references but 1 hypothesis

# current date and time
now = datetime.now()

for folder in os.listdir("data/outputs"):
    try:
         if "output" not in folder:
            continue
         f = open("data/eval-outputs/bleu-%s" %(now), "a")
         for filename in os.listdir("data/outputs/%s" %(folder)):
            try:
                if "txt" not in filename:
                    continue
                reference_file = open("data/outputs/%s/%s" %(folder, filename), 'r')
                reference_lines = reference_file.readlines()

                hypothesis_file = open("data/test.response", 'r')
                hypothesis_lines = hypothesis_file.readlines()
                count = 0
                references = []
                hypothesis = []
                pair_reference = []
                for line in reference_lines:
                    if len(hypothesis_lines) != len(reference_lines):
                        if count%2 != 0:
                            references.append(pair_reference)
                            pair_reference = []

                        pair_reference.append(line.split())
                        count += 1
                    else:
                        references.append([line.split()])

                # print(references)
                for line in hypothesis_lines:
                    hypothesis.append(line.split())

                # print(hypothesis)

                cc = SmoothingFunction()

                bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypothesis,weights = [1], smoothing_function=cc.method3)
                f.write("%s :: %s :: %s \n" %(folder,filename, bleu_score))

            except:
                print(" Failed for given filename %s" %(filename))
         f.write("\n")
    except:
        print("Failed for give folder %s" %(folder))


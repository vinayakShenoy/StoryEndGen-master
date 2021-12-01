from transformers import pipeline
import sys
import os
from datetime import datetime
sentiment_analysis = pipeline("sentiment-analysis")

now = datetime.now()

for folder in os.listdir("data/outputs"):
    try:
         if "output" not in folder:
            continue
         f = open("data/eval-outputs/sentiment-%s" %(now), "a")
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
                positive_sentence_average_sentiment = 0
                postitive_sentence_label = 'POSITIVE'

                negative_sentence_average_sentiment = 0
                negative_sentence_label = 'NEGATIVE'
                result = {}
                for line in reference_lines:
                    if len(hypothesis_lines) != len(reference_lines):
                        if count%2 != 0:
                            #NEGATIVE
                            result = sentiment_analysis(line)[0]
                            if (result['label'] == 'Positive'):
                                negative_sentence_average_sentiment += result['score']
                            else:
                                negative_sentence_average_sentiment -= result['score']
                            print("Sentiment NEGATIVE:: Line: %s :: Label %s :: Score %f" %(line, result['label'], result['score']))

                        else:
                            #POSITIVE
                            result = sentiment_analysis(line)[0]
                            if(result['label'] == 'Positive'):
                                positive_sentence_average_sentiment += result['score']
                            else:
                                positive_sentence_average_sentiment -= result['score']

                            print("Sentiment Positive:: Line: %s :: Label %s :: Score %f" %(line, result['label'], result['score']))


                            # print("Label:", result['label'])
                            # print("Confidence Score:", result['score'])
                            # print()
                        count += 1
                    else:
                        references.append([line.split()])

                pos_op = positive_sentence_average_sentiment/len(hypothesis_lines)
                neg_op = negative_sentence_average_sentiment/len(hypothesis_lines)
                f.write("%s :: %s :: Positive -> %s :: Negative -> %s \n" %(folder,filename, pos_op, neg_op))

            except Exception as e:
                print(e)
                print(" Failed for given filename %s" %(filename))
         f.write("\n")
    except:
        print("Failed for give folder %s" %(folder))

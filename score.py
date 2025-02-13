#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
import json
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(
      description='Score a prediction file using the gold labels.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #  parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('-gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('-pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    parser.add_argument('-no_relation', help='no relation label', default="NA", required=False)
    args = parser.parse_args()
    return args

def score(key, prediction, no_relation, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    
    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
        if gold == no_relation and guess == no_relation:
            pass
        elif gold == no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
        elif gold != no_relation and guess == no_relation:
            gold_by_relation[gold] += 1
        elif gold != no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro

if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = []
    if args.gold_file.endswith('json'):
      with open(args.gold_file, 'r') as myfile:
        data = json.loads(myfile.read())
        for item in data:
          label = item['label']
          key.append(label)
    elif args.gold_file.endswith('tsv'):
      with open(args.gold_file, 'r') as fh:
        for line in fh:
          line = line.strip()
          # Skip empty lines
          if line == "":
            continue
          items = line.split('\t')
          label = items[0]
          key.append(label) 
    #  key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]

    prediction = []
    with open(args.pred_file, 'r') as fh:
      next(fh) # For transformer output
      for line in fh:
        line = line.strip()
        # Skip empty lines
        if line == "":
          continue
        items = line.split('\t')
        label = items[1]
        prediction.append(label)
    #  prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, args.no_relation, verbose=False)

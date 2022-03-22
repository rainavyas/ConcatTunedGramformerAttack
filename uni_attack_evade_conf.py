'''
Perform concatenation adversarial attack on 
GEC system, with aim of finding universal adversarial phrase
that minimises average number of edits between original and 
predicted gec sentence.
'''
import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, count_edits
from Seq2seq import Seq2seq
from eval_uni_attack import set_seeds
import json
from datetime import date
from uni_attack import get_avg, concatenate
from confidence import negative_confidence
from statistics import mean

def is_conf_less_than_thresh(model, sentences, attack_phrase, thresh, delim=''):
    '''
        Return True if the average dataset confidence is less than threshold
    '''
    confs = []
    for sent in sentences:
        sent = concatenate(sent, attack_phrase, delim)
        conf = -1*negative_confidence(sent, model)
        confs.append(conf)
    avg_conf = mean(confs)
    if avg_conf < thresh:
        return True
    return False

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('VOCAB', type=str, help='ASR vocab file')
    commandLineParser.add_argument('LOG', type=str, help='Specify txt file to log iteratively better words')
    commandLineParser.add_argument('--prev_attack', type=str, default='', help='greedy universal attack phrase')
    commandLineParser.add_argument('--num_points', type=int, default=1000, help='Number of training data points to consider')
    commandLineParser.add_argument('--search_size', type=int, default=400, help='Number of words to check')
    commandLineParser.add_argument('--start', type=int, default=0, help='Vocab batch number')
    commandLineParser.add_argument('--conf_thresh', type=float, default=0, help='Confidence Detector threshold')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    commandLineParser.add_argument('--delim', type=str, default='', help='concatenation delimiter')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/uni_attack_evade_conf.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # Load Model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    _, sentences = get_sentences(args.IN, num=args.num_points)

    # Get list of words to try
    with open(args.VOCAB, 'r') as f:
        test_words = json.loads(f.read())
    test_words = [str(word).lower() for word in test_words]

    # Keep only selected batch of words
    start_index = args.start*args.search_size
    test_words = test_words[start_index:start_index+args.search_size]

    # Add blank word at beginning of list
    # test_words = ['']+test_words

    # Initialise empty log file
    with open(args.LOG, 'w') as f:
        f.write("Logged on "+ str(date.today()))

    best = ('none', 1000)
    for word in test_words:
        attack_phrase = args.prev_attack + ' ' + word + '.'
        if not is_conf_less_than_thresh(model, sentences, attack_phrase, args.conf_thresh, delim=args.delim):
            continue
        edits_avg = get_avg(model, sentences, attack_phrase)
        # print(word, edits_avg) # temp debug

        if edits_avg < best[1]:
            best = (word, edits_avg)
            # Write to log
            with open(args.LOG, 'a') as f:
                out = '\n'+best[0]+" "+str(best[1])
                f.write(out)

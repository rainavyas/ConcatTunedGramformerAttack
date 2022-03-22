import sys
import os
import argparse
from gec_tools import get_sentences, correct
from eval_residue_detector import get_best_f_score
from Seq2seq import Seq2seq
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
import math
import numpy as np
from uni_attack import concatenate

def negative_confidence(sentence, model):
    '''
    Calculate negative confidence of sentence using model
    '''
    sf = nn.Softmax(dim=0)
    output_sentence = correct(model, sentence)

    input_ids = model.tokenizer(sentence, return_tensors="pt").input_ids
    all_decoder_input_ids = model.tokenizer(output_sentence, return_tensors="pt").input_ids
    all_decoder_input_ids[0, 0] = model.model.config.decoder_start_token_id
    assert all_decoder_input_ids[0, 0].item() == model.model.config.decoder_start_token_id

    total = 0
    with torch.no_grad():
        for i in range(1, all_decoder_input_ids.size(1)):
            decoder_input_ids = all_decoder_input_ids[:,:i]
            outputs = model.model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
            lm_logits = outputs.logits[:,-1,:].squeeze()
            probs = sf(lm_logits)
            pred_id = all_decoder_input_ids[:,i].squeeze().item()
            prob = probs[pred_id]
            total += math.log(prob)    
    return ((-1)/all_decoder_input_ids.size(1)) * total


if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA', type=str, help='Path to original data file')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('PR', type=str, help='.npz file to save precision recall values')
    commandLineParser.add_argument('--attack_phrase', type=str, default='', help="universal attack phrase")
    commandLineParser.add_argument('--negative', type=str, default='yes', help="yes for negative confidence and no for just confidence")
    commandLineParser.add_argument('--delim', type=str, default='', help='concatenation delimiter')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/confidence.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    multiplier = 1
    if args.negative == 'no':
        multiplier = -1
    
    # Load the GEC model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load the data
    _, orig_sentences = get_sentences(args.DATA)
    adv_sentences = [concatenate(t, args.attack_phrase + '.', delim=args.delim) for t in orig_sentences]

    # Get negative confidence scores
    original_scores = []
    attack_scores = []
    for i, (o,a) in enumerate(zip(orig_sentences, adv_sentences)):
        print(f'On {i}/{len(orig_sentences)}')
        try:
            original_scores.append(multiplier * negative_confidence(o, model))
            attack_scores.append(multiplier * negative_confidence(a, model))
        except:
            print("Failed for ", o)

    labels = [0]*len(original_scores) + [1]*len(attack_scores)
    scores = original_scores + attack_scores
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    best_precision, best_recall, best_f05, best_thresh =  get_best_f_score(precision, recall, thresholds, beta=0.5)
    print(f'Precision: {best_precision}\tRecall: {best_recall}\tF0.5: {best_f05}\tThreshold: {best_thresh}')

    # Save the pr data
    np.savez(args.PR, precision, recall, thresholds)
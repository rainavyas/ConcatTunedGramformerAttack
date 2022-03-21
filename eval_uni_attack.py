'''
Evaluate universal adversarial attack.

Evaluate by counting average number of edits between original input
(with attack phrase) and  GEC model prediction

Also:
- Report how many edits average in original part of sentence and adversarial part.
- Plot histogram of error distibution before and after attack
- Show errant error type distribution before and after attack 
'''

import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, return_edits
from statistics import mean, stdev
# import matplotlib.pyplot as plt
from collections import defaultdict
from Seq2seq import Seq2seq
from uni_attack import concatenate

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_edit_types(edits, edit_type_dict):
    '''
    Increment each type of edit that appears
    '''
    for e in edits:
        edit_type_dict[e.type] += 1

def print_stats(edits, name, original_part_count, adv_part_count):
    print()
    print(f'{name}: {len(edits)} samples')
    edits_mean = mean(edits)
    edits_std = stdev(edits)
    print(f'\nTotal Edits:\tMean: {edits_mean}\t Std: {edits_std}')
    orig_mean = mean(original_part_count)
    orig_std = stdev(original_part_count)
    print(f'\nOriginal Part Edits:\tMean: {orig_mean}\t Std: {orig_std}')
    adv_mean = mean(adv_part_count)
    adv_std = stdev(adv_part_count)
    print(f'\nAdv Part Edits:\tMean: {adv_mean}\t Std: {adv_std}')
    print()

def get_edits_by_part(original_sentence, attack_edits):
    '''
    Determine how many attack edits in which part of attacked sentence
    '''

    edit_strs = [e.o_str for e in attack_edits]
    orig = 0
    adv = 0
    for e_str in edit_strs:
        if original_sentence.find(e_str) == -1:
            adv+=1
        else:
            orig+=1
    return orig, adv


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('FIG', type=str, help='Where to save histogram plot')
    commandLineParser.add_argument('EDIT_TYPE', type=str, help='Path to save edit type information')
    commandLineParser.add_argument('--phrase', type=str, default='', help='Universal adversarial phrase')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_uni_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 
    
    set_seeds(args.seed)
    
    # Load Model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    _, sentences = get_sentences(args.IN)

    # Correction (prediction) for each input sentence and associated number of edits
    edit_counts = []
    edit_counts_with_attack = []
    original_part_count = [] # for adv phrase, count of edits in non adv-part
    adv_part_count = [] # for adv phrase, count of edits in adv-part
    original_edit_types = defaultdict(int)
    adv_edit_types = defaultdict(int)
    for i, sent in enumerate(sentences):

        print(f'On {i}/{len(sentences)}')
        sent_with_attack = sent[:]
        if args.phrase != '':
            attack_phrase = args.phrase + '.'
            sent_with_attack = concatenate(sent, attack_phrase)

        correction = correct(model, sent)
        correction_with_attack = correct(model, sent_with_attack)
        print(f'Input Sentence with attack: {sent_with_attack}')
        print(f'Prediction Sentence without attack: {correction}')
        print(f'Prediction Sentence with attack: {correction_with_attack}')

        edits = return_edits(sent, correction)
        update_edit_types(edits, original_edit_types)
        edits = [e.o_str+' -> '+e.c_str for e in edits]
        edit_counts.append(len(edits))

        edits_with_attack = return_edits(sent_with_attack, correction_with_attack)
        update_edit_types(edits_with_attack, adv_edit_types)
        original_part, adv_part = get_edits_by_part(sent, edits_with_attack)
        original_part_count.append(original_part)
        adv_part_count.append(adv_part)

        edits_with_attack = [e.o_str+' -> '+e.c_str for e in edits_with_attack]
        edit_counts_with_attack.append(len(edits_with_attack))


        print(f'Edits without attack: {edits}')
        print(f'Edits with attack: {edits_with_attack}\n')

    

    # Print stats for all samples
    print_stats(edit_counts_with_attack, 'ALL', original_part_count, adv_part_count)

    # Save edit type distribution to file
    texts = ['Type Original Adv']
    keys1 = list(original_edit_types.keys())
    keys2 = list(adv_edit_types.keys())
    all_keys = sorted(set(keys1+keys2))
    for edit_type in all_keys:
        texts.append(f'\n{edit_type} {original_edit_types[edit_type]} {adv_edit_types[edit_type]}')
    with open(args.EDIT_TYPE, 'w') as f:
            f.writelines(texts)
    
    
    # # Plot histogram of edit count distribution before and after attack
    # plt.hist(edit_counts, bins=12, alpha=0.5, label='No Attack', density=True)
    # plt.hist(edit_counts_with_attack, bins=12, alpha=0.5, label='With Attack', density=True)
    # plt.xlabel("Edits")
    # plt.ylabel("Probability Density")
    # plt.legend()
    # plt.savefig(args.FIG, bbox_inches='tight')
    # plt.clf()








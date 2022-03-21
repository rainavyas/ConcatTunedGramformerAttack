'''
Gives more edit breakdown statistics

Requires the output of eval_error_dist.py at the input
'''

import sys
import os
import argparse

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to edits file')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/more_edit_stats.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the data
    with open(args.IN, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines if len(l)>6]
    
    # Get stats by M, R, U
    M_vals = {'Ref Count':0, 'Pred Total':0, 'Pred Correct':0, 'Pred Insert':0, 'Pred Delete':0}
    R_vals = {'Ref Count':0, 'Pred Total':0, 'Pred Correct':0, 'Pred Insert':0, 'Pred Delete':0}
    U_vals = {'Ref Count':0, 'Pred Total':0, 'Pred Correct':0, 'Pred Insert':0, 'Pred Delete':0}

    all = {'M':M_vals, 'R':R_vals, 'U':U_vals}

    for line in lines:
        elems = line.split()
        try:
            edit_class = elems[0][0]
        except:
            print(line)
        if edit_class in all.keys():
            all[edit_class]['Ref Count'] += int(elems[1])
            all[edit_class]['Pred Total'] += int(elems[2])
            all[edit_class]['Pred Correct'] += int(elems[3])
            all[edit_class]['Pred Insert'] += int(elems[4])
            all[edit_class]['Pred Delete'] += int(elems[5])

    # temp
    refs = [919, 2954, 596]
    for (key, vals), ref in zip(all.items(), refs):
        print(f"{key}:\tRef Count: {vals['Ref Count']/ref:.3}\tPred Total: {vals['Pred Total']/ref:.3}\tPred Correct: {vals['Pred Correct']/ref:.3}\tPred Insert: {vals['Pred Insert']/ref:.3}\tPred Delete: {vals['Pred Delete']/ref:.3}")

        
    # print stats
    # for key, vals in all.items():
    #     print(f"{key}:\tRef Count: {vals['Ref Count']}\tPred Total: {vals['Pred Total']}\tPred Correct: {vals['Pred Correct']}\tPred Insert: {vals['Pred Insert']}\tPred Delete: {vals['Pred Delete']}")

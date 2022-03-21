import random
import errant

random.seed(10)

def get_sentences(data_path, num=-1):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    if num > 0:
        print("Here Type ", type(num))
        random.shuffle(lines)
        lines = lines[:num]
    texts = [' '.join(l.rstrip('\n').split()[1:]) for l in lines]
    ids = [l.rstrip('\n').split()[0] for l in lines]

    # Remove space before full stops at end
    texts = [t[:-2]+'.' if t[-2:]==' .' else t for t in texts]

    return ids, texts

def correct(model, sentence):

    '''Gramformer decoding'''

    result = model.correct(sentence, max_candidates=1)[0]
    return result

def count_edits(input, prediction):
    '''
    Count number of edits
    '''
    annotator = errant.load('en')
    input = annotator.parse(input)
    prediction = annotator.parse(prediction)
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    return len(edits)

def return_edits(input, prediction):
    '''
    Get edits
    '''
    annotator = errant.load('en')
    input = annotator.parse(input)
    prediction = annotator.parse(prediction)
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    for e in edits:
        e = annotator.classify(e)
    return edits
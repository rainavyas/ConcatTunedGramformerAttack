'''
This script is inspired from Edie's class definition of Seq2seq
.correct function inspired from https://github.com/PrithivirajDamodaran/Gramformer/blob/main/gramformer/gramformer.py

'''
import torch.nn as nn

# huggingface api
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import warnings
warnings.filterwarnings("ignore")


class Seq2seq(nn.Module):

    """ T5 enc-dec model """

    def __init__(self):

        super(Seq2seq, self).__init__()

        model_name = "prithivida/grammar_error_correcter_v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def correct(self, input_sentence, max_candidates=1):
        correction_prefix = "gec: "
        input_sentence = correction_prefix + input_sentence
        input_ids = self.tokenizer.encode(input_sentence, return_tensors='pt')

        preds = self.model.generate(
            input_ids,
            do_sample=True, 
            max_length=128, 
            top_k=50, 
            top_p=0.95, 
            early_stopping=True,
            num_return_sequences=max_candidates)

        corrected = set()
        for pred in preds:  
          corrected.add(self.tokenizer.decode(pred, skip_special_tokens=True).strip())

        corrected = list(corrected)
        return corrected

        

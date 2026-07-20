# tokenizer.py
all_words = ['0','1','2','3','4','5','6','7','8','9','+','-','*','=','<EOS>']

vocab_size = len(all_words)

vocab = {token:integer for integer, token in enumerate(all_words)}

import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
        
    def encode(self, text):
        #ids  = [self.str_to_int[s] for s in list(text)]
        ids = [self.str_to_int[c] for c in text]
        ids.append(self.str_to_int["<EOS>"])
        return ids
    
    #def decode(self, ids):
    #    text = "".join(self.int_to_str[i] for i in ids)
    #    return text
    def decode(self, ids):
        text = []
        for i in ids:
            token = self.int_to_str[i]
            if token == "<EOS>":
                break
            text.append(token)
        return "".join(text)

if __name__=="__main__":
    print("vocab:", vocab)
    print("vocab_size:", vocab_size)
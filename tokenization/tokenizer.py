import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [ 
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int.get(s, self.str_to_int.get("<|unk|>")) for s in preprocessed]
        return ids
    
    def decode(self, token_id):
        text = " ".join([self.int_to_str[i] for i in token_id])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

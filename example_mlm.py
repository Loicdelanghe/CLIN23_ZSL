from transformers import AutoTokenizer
import torch
from prompt import Prompting


# DEMO sentiment

model_path = "GroNLP/bert-base-dutch-cased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompting = Prompting(model=model_path)

text_piece = "Het eten in dit restaurant is heel lekker."
fulltext = text_piece + " " + "Het sentiment van deze review is [MASK]."


out = prompting.compute_tokens_prob(fulltext,
                                    token_list1=['positief'],
                                    token_list2=['negatief'])

print(out)

# DEMO information extraction

text_piece = "Parijs is prachtig in december"
unigrams = text_piece.split(" ")

label_list = ['locatie', 'tijdstip']

for word in unigrams:
    full_text = '%s is een [MASK].' % word
    out = prompting.compute_tokens_prob(full_text,
                                        token_list1=[label_list[0]],
                                        token_list2=[label_list[1]])

    t_idx, t_m = torch.argmax(out), torch.max(out)
    if t_m > 0.95:
        formatted_output = 'TOKEN: %s \nPREDICTED AS: %s \nCERTAINTY: %f\n' % (word, label_list[t_idx], float(t_m))
        print(formatted_output)

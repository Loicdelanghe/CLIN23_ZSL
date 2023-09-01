from transformers import pipeline
import pprint


classifier = pipeline(
                      task="zero-shot-classification",
                      model='LoicDL/bert-base-dutch-cased-finetuned-snli'
                    )


text_piece = "Het eten in dit restaurant is heel lekker."
labels = ["Positief", "Negatief", "Neutraal"]
template = "Het sentiment van deze review is {}"

predictions = classifier(text_piece,
           labels,
           multi_class=False,
           hypothesis_template=template
           )

pprint.pprint(predictions)
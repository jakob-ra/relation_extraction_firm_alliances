# import pandas as pd
from spacy.cli.project.assets import project_assets
from pathlib import Path
from spacy.cli.project.run import project_run
import spacy
import random
import typer
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from scripts.rel_pipe import make_relation_extractor, score_relations
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

# We load the relation extraction (REL)
nlp_rel = spacy.load('training/model-best', vocab=nlp.vocab)

nlp = spacy.load('en_core_web_trf', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'], vocab=nlp_rel.vocab)
nlp.add_pipe('sentencizer', after='transformer')
nlp.add_pipe('organization_extractor', after='ner')

nlp.add_pipe('transformer', name='rel_transformer', source=nlp_rel)
nlp.add_pipe('relation_extractor', source=nlp_rel)
print(nlp.component_names)

# nlp = spacy.load('training/model-best')

text = ['Microsoft Inc and Sun Microsystems just announced a new strategic alliance to jointly research'
      'cloud computing infrastructure. Barack Obama mentioned something else.']

for doc in nlp.pipe(text):
    print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")
    for entry in doc._.rel.values():
        print([x for x in entry.items() if x[1] >= 0.9])
        [x for x in entry.items()]

# text = ['Microsoft Inc and Sun Microsystems just announced that they will break up.']
#
# kb = pd.io.json.read_json(path_or_buf='/Users/Jakob/Documents/Thomson_SDC/Full/SDC_training_dict.json',
#                           orient='records', lines=True)
#
#
# sdc = pd.read_pickle('/Users/Jakob/Documents/Thomson_SDC/Full/SDC_Strategic_Alliances_Full.pkl')
#
#
# def test_rel_project():
#     root = Path(__file__).parent
#     project_assets(root)
#     project_run(root, "all", capture=True)
#
#
# if __name__ == "__main__":
#     project_run(project_dir=Path.cwd(), subcommand='train_cpu')



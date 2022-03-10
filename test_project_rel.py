import pandas as pd
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
nlp_rel = spacy.load('training/model-best')

nlp = spacy.load('en_core_web_trf', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'], vocab=nlp_rel.vocab)
nlp.add_pipe('sentencizer', after='transformer')
nlp.add_pipe('organization_extractor', after='ner')

nlp.add_pipe('transformer', name='rel_transformer', source=nlp_rel)
nlp.add_pipe('relation_extractor', source=nlp_rel)
print(nlp.component_names)

# nlp = spacy.load('training/model-best')

texts = ['Microsoft Inc and Sun Microsystems just announced a new strategic alliance to jointly research'
      'cloud computing infrastructure.', 'Barack Obama mentioned something else.', 'Clark Development announced '
      'a new partnership with BlissCo.', 'BHP Hilton Group just closed a licensing deal with SF Airlines.']




def extract_relations(texts, threshold=0.9):
    results = []
    for doc in nlp.pipe(texts):
        print(doc.text)
        doc_res = {}
        res = {}
        for ent_pair in doc._.rel:
            entry = doc._.rel[ent_pair]
            relations = set([rel_type for rel_type in entry if entry[rel_type] >= threshold])
            firms = [e.text for e in doc.ents if e.start in ent_pair]
            if frozenset(firms) in doc_res: # if already relations detected for a firm pair, add to them
                existing_relations = doc_res[frozenset(firms)]
                doc_res[frozenset(firms)] = existing_relations | relations
            else:
                doc_res[frozenset(firms)] = relations
        print(doc_res)
        results.append(doc_res)


    return results

extract_relations(texts, threshold=0.9)

print('Downloading test file...')
df = pd.read_pickle('https://www.dropbox.com/s/36a07va701ap6h0/lexisnexis_firm_alliances_2017_cleaned_min_2_companies.pkl?dl=1')
print('Done!')

import time

t0 = time.time()

sample_size = 10
test = df.sample(sample_size)
results = extract_relations(test.text, threshold=0.9)
results = pd.Series(results)
test = test.merge(results, left_index=True, right_index=True)

t1 = time.time()
print(f'Time elapsed for processing {sample_size} documents: {t1-t0}')



# kb = pd.io.json.read_json(path_or_buf='https://drive.google.com/uc?id=1yp5VQwbvv9xVZ__l5PsHo1TbNqI8P6I3&export=download',
#                           orient='records', lines=True)
# test = kb[kb.meta.apply(lambda x: x['split'] == 'test')]
#
# test = test.sample(100)
#
# import time
#
# start = time.time()
# test['pred_relationships'] = test.document.apply(lambda x: nlp(x)._.rel)
# end = time.time()
# print(f'Elapsed time: {end-start} seconds.')
#
#
# # read firm list
# path = 'C:/Users/Jakob/Documents/Orbis/Full/BvD_ID_and_Name.txt'
# orbis = pd.read_csv(path, sep='\t')
#
# firm_names =
#
# # entity ruler
# import spacy
#
# nlp = spacy.blank('en')
# ruler = nlp.add_pipe('entity_ruler')
# patterns = [{"label": "ORG", "pattern": [{"LOWER": "apple inc"}, {"LOWER": "apple computers"}], "id": "Apple - bvdid"}]
# ruler.add_patterns(patterns)
# doc = nlp('Apple inc introduced a new processor.')
# ents = [(ent.text, ent.label_) for ent in doc.ents]
# ents

# print(test.pred_relationships.values)
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



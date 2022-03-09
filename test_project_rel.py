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
      'cloud computing infrastructure. Barack Obama mentioned something else.', 'Clark Development announced '
      'a new partnership with BlissCo.']

for doc in nlp.pipe(texts):
    print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")
    for entry in doc._.rel.values():
        print([x for x in entry.items() if x[1] >= 0.9])
        [x for x in entry.items()]

df = pd.read_pickle('C:/Users/Jakob/Documents/lexisnexis_firm_alliances_combined_new.pkl')

# keep only year 2017
df = df[df.publication_date.dt.year == 2017]

# keep only english
df = df[df.lang == 'en']

# combine title and content
df['text'] = df.title + '. ' + df.content


from nltk.tokenize import sent_tokenize
df['sentences'] = df.text.apply(sent_tokenize)
df['num_sentences'] = df.sentences.apply(len)


df.sentences.sample().values
df.num_sentences.describe()

# cut off docs at maximum length
max_len = 20
df['sentences'] = df.sentences.apply(lambda x: x[:max_len])
df['text'] = df.sentences.apply(lambda x: ' '.join(x))
df.drop(columns=['sentences'], inplace=True)

# focus on news mentioning at least 2 companies
df = df[df.company.str.len() > 1]


def extract_relations(texts, threshold=0.9):
    results = []
    for doc in nlp.pipe(texts):
        doc_res = {}
        for ent_pair in doc._.rel:
            entry = doc._.rel[ent_pair]
            relations = [x for x in entry if entry[x] >= threshold]
            firms = [e.text for e in doc.ents if e.start in ent_pair]
            res = {frozenset(firms): set(relations)}
        doc_res.update(res) # issue: if a->b is detected as SA but not b->a then this will overwrite and save as no relation
        results.append(doc_res)

    return results


test = df.sample(1000)
results = extract_relations(test.text, threshold=0.6)
results = pd.Series(results)
test = test.merge(results, left_index=True, right_index=True)






# focus on news about JVs, alliances, and agreements
# df.subject.explode().value_counts().head(50)
# df[df.subject.apply(lambda x: [sub for sub in x if sub in ['JOINT VENTURES', 'ALLIANCES & PARTNERSHIPS', 'AGREEMENTS']]).str.len() > 0]



#
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



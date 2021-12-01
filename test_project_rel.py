from spacy.cli.project.assets import project_assets
from pathlib import Path
from spacy.cli.project.run import project_run
import spacy
import random
import typer
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

# We load the relation extraction (REL)
nlp = spacy.load("training/model-best")

# We take the entities generated from the NER pipeline and input them to the REL pipeline
for name, proc in nlp.pipeline:
        doc = proc(doc)# Here, we split the paragraph into sentences and apply the relation extraction for each pair of entities found in each sentence.for value, rel_dict in doc._.rel.items():
        for sent in doc.sents:
            for e in sent.ents:
                for b in sent.ents:
                    if e.start == value[0] and b.start == value[1]:
                        if rel_dict['s'] >=0.9:
                            print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")

def test_rel_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", capture=True)


if __name__ == "__main__":
    project_run(project_dir=Path.cwd(), subcommand='train_cpu')



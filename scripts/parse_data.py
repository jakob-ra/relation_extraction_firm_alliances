import json

import typer
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

import spacy

msg = Printer()

SYMM_LABELS =  ['StrategicAlliance', 'JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment',
       'Licensing', 'Supply', 'Exploration', 'TechnologyTransfer']
MAP_LABELS = {
    'StrategicAlliance': 'Strategic Alliance',
    'JointVenture': 'Joint Venture',
    'Marketing': 'Marketing agreement',
    'Manufacturing': 'Manufacturing agreement',
    'ResearchandDevelopment': 'Research and Development agreement',
    'Licensing': 'Licensing agreement',
    'Supply': 'Supply agreement',
    'Exploration': 'Exploration agreement',
    'TechnologyTransfer': 'Technology Transfer'
}

nlp = spacy.blank("en")

def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Thomson SDC annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": []}
    ids = {"train": set(), "dev": set(), "test": set()}
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}

    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            neg = 0
            pos = 0
            try:
                # Parse the tokens
                tokens = nlp(example["document"])
                spaces = []
                spaces = [True if tok.whitespace_ else False for tok in tokens]
                words = [t.text for t in tokens]
                doc = Doc(nlp.vocab, words=words, spaces=spaces)

                # Parse the entities
                spans = example["tokens"]
                entities = []
                span_end_to_start = {}
                for span in spans:
                    entity = doc.char_span(
                        span["start"], span["end"], label=span["entityLabel"]
                    )
                    span_end_to_start[span["token_start"]] = span["token_start"]
                    entities.append(entity)
                    span_starts.add(span["token_start"])
                doc.ents = entities

                # Parse the relations
                rels = {}
                for x1 in span_starts:
                    for x2 in span_starts:
                        rels[(x1, x2)] = {}
                relations = example["relations"]
                for relation in relations:
                    # the 'head' and 'child' annotations refer to the end token in the span
                    # but we want the first token
                    start = span_end_to_start[relation["head"]]
                    end = span_end_to_start[relation["child"]]
                    label = relation["relationLabel"]
                    label = MAP_LABELS[label]
                    if label not in rels[(start, end)]:
                        rels[(start, end)][label] = 1.0
                        pos += 1
                    if label in SYMM_LABELS:
                        if label not in rels[(end, start)]:
                            rels[(end, start)][label] = 1.0
                            pos += 1

                # The annotation is complete, so fill in zero's where the data is missing
                for x1 in span_starts:
                    for x2 in span_starts:
                        for label in MAP_LABELS.values():
                            if label not in rels[(x1, x2)]:
                                neg += 1
                                rels[(x1, x2)][label] = 0.0
                doc._.rel = rels

                # only keeping documents with at least 1 positive case
                if pos > 0:
                    article_id = example["meta"]["source"]
                    article_id = article_id.split("Deal Number ")[1]
                    split = example["meta"]["split"]
                    if split == 'dev':
                        ids["dev"].add(article_id)
                        docs["dev"].append(doc)
                        count_pos["dev"] += pos
                        count_all["dev"] += pos + neg
                    elif split == 'test':
                        ids["test"].add(article_id)
                        docs["test"].append(doc)
                        count_pos["test"] += pos
                        count_all["test"] += pos + neg
                    elif split == 'train':
                        ids["train"].add(article_id)
                        docs["train"].append(doc)
                        count_pos["train"] += pos
                        count_all["train"] += pos + neg
                    else:
                        msg.fail('Skipping doc because it is not specified as part of the train, test, or dev set.')
            except KeyError as e:
                msg.fail(f"Skipping doc because of key error: {e} in {example['meta']['source']}")

    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences from {len(ids['train'])} articles, "
        f"{count_pos['train']}/{count_all['train']} pos instances."
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences from {len(ids['dev'])} articles, "
        f"{count_pos['dev']}/{count_all['dev']} pos instances."
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test sentences from {len(ids['test'])} articles, "
        f"{count_pos['test']}/{count_all['test']} pos instances."
    )


if __name__ == "__main__":
    typer.run(main)

# json_loc = "C:/Users/Jakob/Documents/GitHub/relation_extraction_firm_alliances/assets/SDC_training_dict.json"
# train_file = "C:/Users/Jakob/Documents/GitHub/relation_extraction_firm_alliances/assets/SDC_training_dict.json"
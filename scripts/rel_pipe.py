from itertools import islice
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from spacy.scorer import PRFScore
from spacy.scorer import Scorer
from thinc.types import Floats2d
import numpy
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer
import re
import unidecode

Doc.set_extension("rel", default={}, force=True)
msg = Printer()


def firm_name_clean(firm_name, lower=True, remove_punc=True, remove_legal=True, remove_parentheses=True):
    # make string
    firm_name = str(firm_name)
    firm_name = unidecode.unidecode(firm_name)
    # lowercase
    if lower:
        firm_name = firm_name.lower()
    # remove punctuation
    if remove_punc:
        firm_name = firm_name.translate(str.maketrans('', '', '!"#$%\\\'*+,./:;<=>?@^_`{|}~'))
    # remove legal identifiers
    if remove_legal:
        legal_identifiers = ["co", "inc", "ag", "ltd", "lp", "llc", "pllc", "llp", "plc", "ltdplc", "corp",
                             "corporation", "ab", "cos", "cia", "sa", "company", "companies", "consolidated",
                             "stores", "limited", "srl", "kk", "gmbh", "pty", "group", "yk", "bhd",
                             "limitada", "holdings", "kg", "bv", "pte", "sas", "ilp", "nl", "genossenschaft",
                             "gesellschaft", "aktiengesellschaft", "ltda", "nv", "oao", "holding", "se",
                             "oy", "plcnv", "the", "neft", "& co", "&co"]
        pattern = '|'.join(legal_identifiers)
        pattern = '\\b(' + pattern + ')\\b'  # match only word boundaries
        firm_name = re.sub(pattern, '', firm_name)
    # remove parentheses and anything in them: Bayerische Motoren Werke (BMW) -> Bayerische Motoren Werke
    if remove_parentheses:
        firm_name = re.sub(r'\([^()]*\)', '', firm_name)

    # make hyphens consistent
    firm_name = firm_name.replace(' - ', '-')

    # remove ampersand symbol
    firm_name = firm_name.replace('&amp;', '&')
    firm_name = firm_name.replace('&amp', '&')

    # strip
    firm_name = firm_name.strip()

    return firm_name

@Language.component("organization_extractor")
def organization_extractor(doc):
    doc.ents = tuple([e for e in doc.ents if e.label_ == 'ORG'])

    return doc

import pandas as pd
print('Downloading firm lookup list...')
firm_lookup_list = pd.read_csv('https://www.dropbox.com/s/bsq4m09j3ovqsy1/firm_lookup_list.csv.gzip?dl=1', compression='gzip')
firm_lookup_list = set(firm_lookup_list.company.to_list())
print('Done!')

@Language.component("firm_name_lookup") # only keeps a recognized entity if it is in the firm name lookup table
def organization_extractor(doc):
    print(doc.ents)
    doc.ents = tuple([e for e in doc.ents if firm_name_clean(e.text) in firm_lookup_list])
    print(doc.ents)
    return doc

@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
    },
)
def make_relation_extractor(
    nlp: Language, name: str, model: Model, *, threshold: float
):
    """Construct a RelationExtractor component."""
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)


class RelationExtractor(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        if not isinstance(label, str):
            raise ValueError("Only strings can be added as labels to the RelationExtractor")
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        # check that there are actually any candidate instances in this batch of examples
        total_instances = len(self.model.attrs["get_instances"](doc))
        if total_instances <= 1:
            # msg.info("Could not determine more than one instance in doc - returning doc as is.")
            return doc

        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)

        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        get_instances = self.model.attrs["get_instances"]
        total_instances = sum([len(get_instances(doc)) for doc in docs])
        if total_instances <= 1:
            msg.info("Could not determine more than one instance in any docs - can not make any predictions.")
            scores = numpy.zeros((total_instances, len(self.labels)), dtype="f")
            return scores
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        c = 0
        get_instances = self.model.attrs["get_instances"]
        for doc in docs:
            for (e1, e2) in get_instances(doc):
                offset = (e1.start, e2.start)
                if offset not in doc._.rel:
                    doc._.rel[offset] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[offset][label] = scores[c, j]
                c += 1

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # check that there are actually any candidate instances in this batch of examples
        total_instances = 0
        for eg in examples:
            total_instances += len(self.model.attrs["get_instances"](eg.predicted))
        if total_instances <= 1:
            # msg.info("Less than two instances in doc.")
            return losses

        # run the model
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths = self._examples_to_truth(examples)
        gradient = scores - truths
        mean_square_error = (gradient ** 2).sum(axis=1).mean()
        return float(mean_square_error), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                relations = example.reference._.rel
                for indices, label_dict in relations.items():
                    for label in label_dict.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError("Call begin_training with relevant entities and relations annotated in "
                             "at least a few reference examples!")
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(self, examples: List[Example]) -> Optional[numpy.ndarray]:
        # check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(self.model.attrs["get_instances"](eg.reference))
        if nr_instances <= 1:
            print("less than two instances, returning None")
            return None

        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        c = 0
        for i, eg in enumerate(examples):
            for (e1, e2) in self.model.attrs["get_instances"](eg.reference):
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1

        truths = self.model.ops.asarray(truths)
        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        res = score_relations(examples, self.threshold)

        return res


def score_relations(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score a batch of examples."""
    micro_prf = PRFScore()
    # print(f'Number of examples: {len(list(examples))}')
    # get labels from first example
    for example in examples:
        labels = list(list(example.reference._.rel.values())[0].keys())
        break
    f_per_type = {label: PRFScore() for label in labels}
    for example in examples:
        gold = example.reference._.rel
        pred = example.predicted._.rel
        for key, pred_dict in pred.items():
            gold_labels = [k for (k, v) in gold.get(key, {}).items() if v == 1.0]
            for k, v in pred_dict.items():
                if v >= threshold:
                    if k in gold_labels:
                        micro_prf.tp += 1
                        f_per_type[k].tp += 1
                    else:
                        micro_prf.fp += 1
                        f_per_type[k].fp += 1
                else:
                    if k in gold_labels:
                        micro_prf.fn += 1
                        f_per_type[k].fn += 1

    scores = {
        "rel_micro_p": micro_prf.precision,
        "rel_micro_r": micro_prf.recall,
        "rel_micro_f": micro_prf.fscore
    }

    for label in labels:
        label_name = label.replace(' ', '_').lower()
        scores["rel_f_" + label_name] = f_per_type[label].to_dict()['f']

    return scores




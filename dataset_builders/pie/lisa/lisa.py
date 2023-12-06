import os.path
from collections import defaultdict

from pie_datasets.builders import BratBuilder
from pie_datasets.builders.brat import BratDocument
from pie_datasets.core.dataset import DocumentConvertersType
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

URL = os.path.abspath("data/KEEPHA_lifeline_de_batch_1.zip")
SPLIT_PATHS = {"train": "KEEPHA_lifeline_de_batch_1"}


def convert_to_text_document_with_labeled_spans_and_binary_relations(
    document: BratDocument,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    new_spans = []
    new_rels = []
    old2new_span = defaultdict(list)

    spans = document.spans
    new_doc = TextDocumentWithLabeledSpansAndBinaryRelations(
        text=document.text, id=document.id, metadata=document.metadata
    )

    candidate_spans = [span for span in spans if len(span.slices) > 1]

    for span in spans:
        slices = span.slices
        label = span.label
        score = span.score
        for slice in slices:
            new_span = LabeledSpan(start=slice[0], end=slice[1], label=label, score=score)
            new_spans.append(new_span)
            old2new_span[span].append(new_span)

    new_doc.labeled_spans.extend(new_spans)

    for span in candidate_spans:
        new_spans = old2new_span[span]
        new_spans.sort(key=lambda span: span.start)
        new_rels.extend(
            [
                BinaryRelation(
                    head=new_spans[i], tail=new_spans[i + 1], label="parts_of_same", score=1.0
                )
                for i in range(len(new_spans) - 1)
            ]
        )

    rels = document.relations

    for rel in rels:
        head = rel.head
        tail = rel.tail

        new_heads = old2new_span[head]
        new_tails = old2new_span[tail]

        for new_head in new_heads:
            for new_tail in new_tails:
                new_rel = BinaryRelation(
                    head=new_head, tail=new_tail, label=rel.label, score=rel.score
                )
                new_rels.append(new_rel)

    new_doc.binary_relations.extend(new_rels)

    return new_doc


class Lisa(BratBuilder):

    BASE_DATASET_PATH = "DFKI-SLT/brat"
    BASE_DATASET_REVISION = "052163d34b4429d81003981bc10674cef54aa0b8"

    # we need to add None to the list of dataset variants to support the default dataset variant
    BASE_BUILDER_KWARGS_DICT = {
        dataset_variant: {"url": URL, "split_paths": SPLIT_PATHS}
        for dataset_variant in ["default", "merge_fragmented_spans", None]
    }

    @property
    def document_converters(self) -> DocumentConvertersType:
        if self.config.name == "default":
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: convert_to_text_document_with_labeled_spans_and_binary_relations
            }
        elif self.config.name == "merge_fragmented_spans":
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: {
                    "spans": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        else:
            raise ValueError(f"Unknown dataset variant: {self.config.name}")

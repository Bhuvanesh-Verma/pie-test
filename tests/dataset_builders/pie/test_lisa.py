import dataclasses
from collections import defaultdict
from typing import List, Optional, Union

import pytest
from datasets import disable_caching
from pie_datasets import DatasetDict
from pie_datasets.builders.brat import BratDocument, BratDocumentWithMergedSpans
from pie_modules.document.processing import tokenize_document
from pytorch_ie.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations, TokenBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset_builders.pie.lisa.lisa import Lisa
from tests.dataset_builders import PIE_BASE_PATH


@dataclasses.dataclass
class TestTokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TestTokenDocumentWithLabeledSpansAndBinaryRelations(TestTokenDocumentWithLabeledSpans):
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


disable_caching()

DATASET_NAME = "lisa"
PIE_DATASET_PATH = PIE_BASE_PATH / DATASET_NAME
SPLIT_SIZES = {"train": 118}


@pytest.fixture(scope="module", params=["default", "merge_fragmented_spans"])
def dataset_variant(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def dataset(dataset_variant) -> DatasetDict:
    return DatasetDict.load_dataset(str(PIE_DATASET_PATH), name=dataset_variant)


def test_dataset(dataset):
    assert dataset is not None
    assert {name: len(ds) for name, ds in dataset.items()} == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset) -> Union[BratDocument, BratDocumentWithMergedSpans]:
    return dataset["train"][0]


def test_document(document, dataset_variant):
    assert document is not None
    assert document.text.startswith(
        "Hallo liebe <user>, danke für deine lieben Worte, hier im Forum sind alle so nett und hilfsbereit."
    )
    if dataset_variant == "default":
        assert isinstance(document, BratDocument)
        span = document.spans[0]
        assert isinstance(span, LabeledMultiSpan)
        assert len(document.spans) == 10
        # sort the entities by their start position and convert them to tuples

        sorted_entity_tuples = [
            (
                tuple(ent.target[slice_start:slice_end] for slice_start, slice_end in ent.slices),
                ent.label,
            )
            for ent in sorted(document.spans, key=lambda ent: ent.slices[0][0])
        ]
        # Checking all entities
        assert sorted_entity_tuples == [
            (("Brennen",), "DISORDER"),
            (("massive Nervosität",), "DISORDER"),
            (("Citalopram",), "DRUG"),
            (("Angstpatienten",), "DISORDER"),
            (("AD",), "DRUG"),
            (("Citalopram",), "DRUG"),
            (("nicht mehr nehme",), "CHANGE_TRIGGER"),
            (("gehts mir wieder viel besser",), "OPINION"),
            (("schreckliche Überdrehtheit",), "DISORDER"),
            (("dachte echt ich werde verrückt",), "OPINION"),
        ]

        # check the relations
        assert len(document.relations) == 6

        relation_tuples = [
            (
                tuple(
                    rel.head.target[slice_start:slice_end]
                    for slice_start, slice_end in rel.head.slices
                ),
                rel.label,
                tuple(
                    rel.tail.target[slice_start:slice_end]
                    for slice_start, slice_end in rel.tail.slices
                ),
            )
            for rel in document.relations
        ]
        assert relation_tuples == [
            (("nicht mehr nehme",), "SIGNALS_CHANGE_OF", ("Citalopram",)),
            (("Citalopram",), "CAUSED", ("massive Nervosität",)),
            (("massive Nervosität",), "CAUSED", ("Brennen",)),
            (("Citalopram",), "CAUSED", ("schreckliche Überdrehtheit",)),
            (("gehts mir wieder viel besser",), "IS_OPINION_ABOUT", ("Citalopram",)),
            (
                ("dachte echt ich werde verrückt",),
                "IS_OPINION_ABOUT",
                ("schreckliche Überdrehtheit",),
            ),
        ]
    elif dataset_variant == "merge_fragmented_spans":
        assert isinstance(document, BratDocumentWithMergedSpans)
        span = document.spans[0]
        assert isinstance(span, LabeledSpan)

        assert len(document.spans) == 10
        # sort the entities by their start position and convert them to tuples

        sorted_entity_tuples = [
            (str(ent), ent.label) for ent in sorted(document.spans, key=lambda ent: ent.start)
        ]
        # Checking all entities
        assert sorted_entity_tuples == [
            ("Brennen", "DISORDER"),
            ("massive Nervosität", "DISORDER"),
            ("Citalopram", "DRUG"),
            ("Angstpatienten", "DISORDER"),
            ("AD", "DRUG"),
            ("Citalopram", "DRUG"),
            ("nicht mehr nehme", "CHANGE_TRIGGER"),
            ("gehts mir wieder viel besser", "OPINION"),
            ("schreckliche Überdrehtheit", "DISORDER"),
            ("dachte echt ich werde verrückt", "OPINION"),
        ]

        # check the relations
        assert len(document.relations) == 6

        relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in document.relations]
        assert relation_tuples == [
            ("nicht mehr nehme", "SIGNALS_CHANGE_OF", "Citalopram"),
            ("Citalopram", "CAUSED", "massive Nervosität"),
            ("massive Nervosität", "CAUSED", "Brennen"),
            ("Citalopram", "CAUSED", "schreckliche Überdrehtheit"),
            ("gehts mir wieder viel besser", "IS_OPINION_ABOUT", "Citalopram"),
            ("dachte echt ich werde verrückt", "IS_OPINION_ABOUT", "schreckliche Überdrehtheit"),
        ]

    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset, dataset_variant
) -> Optional[DatasetDict]:
    if dataset_variant not in ["default", "merge_fragmented_spans"]:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
    converted_dataset = dataset.to_document_type(TextDocumentWithLabeledSpansAndBinaryRelations)

    return converted_dataset


def test_dataset_of_text_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, dataset_variant
):
    if dataset_variant == "default":
        # Check that the conversion is correct and the data makes sense
        # get a document to check
        doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
        assert isinstance(doc, TextDocumentWithLabeledSpansAndBinaryRelations)
        # check the entities
        assert len(doc.labeled_spans) == 10
        # sort the entities by their start position and convert them to tuples

        sorted_entity_tuples = [
            (str(ent), ent.label) for ent in sorted(doc.labeled_spans, key=lambda ent: ent.start)
        ]
        # Checking all entities
        assert sorted_entity_tuples == [
            ("Brennen", "DISORDER"),
            ("massive Nervosität", "DISORDER"),
            ("Citalopram", "DRUG"),
            ("Angstpatienten", "DISORDER"),
            ("AD", "DRUG"),
            ("Citalopram", "DRUG"),
            ("nicht mehr nehme", "CHANGE_TRIGGER"),
            ("gehts mir wieder viel besser", "OPINION"),
            ("schreckliche Überdrehtheit", "DISORDER"),
            ("dachte echt ich werde verrückt", "OPINION"),
        ]

        # check the relations
        assert len(doc.binary_relations) == 6

        relation_tuples = [
            (str(rel.head), rel.label, str(rel.tail)) for rel in doc.binary_relations
        ]
        assert relation_tuples == [
            ("nicht mehr nehme", "SIGNALS_CHANGE_OF", "Citalopram"),
            ("Citalopram", "CAUSED", "massive Nervosität"),
            ("massive Nervosität", "CAUSED", "Brennen"),
            ("Citalopram", "CAUSED", "schreckliche Überdrehtheit"),
            ("gehts mir wieder viel besser", "IS_OPINION_ABOUT", "Citalopram"),
            ("dachte echt ich werde verrückt", "IS_OPINION_ABOUT", "schreckliche Überdrehtheit"),
        ]

        # Document with multiple span entity
        doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][-1]
        assert isinstance(doc, TextDocumentWithLabeledSpansAndBinaryRelations)
        # check the entities
        assert len(doc.labeled_spans) == 9
        # sort the entities by their start position and convert them to tuples

        sorted_entity_tuples = [
            (str(ent), ent.label) for ent in sorted(doc.labeled_spans, key=lambda ent: ent.start)
        ]
        # Checking all entities
        assert sorted_entity_tuples == [
            ("wieder", "CHANGE_TRIGGER"),
            ("Cipralex", "DRUG"),
            ("bekommen", "CHANGE_TRIGGER"),
            ("5 mg", "MEASURE"),
            ("fast alle Nebenwirkungen", "DISORDER"),
            ("generalisierte Angst", "DISORDER"),
            ("Angst vor allem möglichen", "DISORDER"),
            ("Nebenwirkungen", "DISORDER"),
            ("Arzt", "DOCTOR"),
        ]

        # check the relations
        assert len(doc.binary_relations) == 6

        relation_tuples = [
            (str(rel.head), rel.label, str(rel.tail)) for rel in doc.binary_relations
        ]
        assert relation_tuples == [
            ("wieder", "parts_of_same", "bekommen"),
            ("Cipralex", "HAS_DOSAGE", "5 mg"),
            ("wieder", "SIGNALS_CHANGE_OF", "Cipralex"),
            ("bekommen", "SIGNALS_CHANGE_OF", "Cipralex"),
            ("Cipralex", "CAUSED", "fast alle Nebenwirkungen"),
            ("Angst vor allem möglichen", "REFERS_TO", "generalisierte Angst"),
        ]
    elif dataset_variant == "merge_fragmented_spans":
        # Check that the conversion is correct and the data makes sense
        # get a document to check
        doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
        assert isinstance(doc, TextDocumentWithLabeledSpansAndBinaryRelations)
        # check the entities
        assert len(doc.labeled_spans) == 10
        # sort the entities by their start position and convert them to tuples

        sorted_entity_tuples = [
            (str(ent), ent.label) for ent in sorted(doc.labeled_spans, key=lambda ent: ent.start)
        ]
        # Checking all entities
        assert sorted_entity_tuples == [
            ("Brennen", "DISORDER"),
            ("massive Nervosität", "DISORDER"),
            ("Citalopram", "DRUG"),
            ("Angstpatienten", "DISORDER"),
            ("AD", "DRUG"),
            ("Citalopram", "DRUG"),
            ("nicht mehr nehme", "CHANGE_TRIGGER"),
            ("gehts mir wieder viel besser", "OPINION"),
            ("schreckliche Überdrehtheit", "DISORDER"),
            ("dachte echt ich werde verrückt", "OPINION"),
        ]

        # check the relations
        assert len(doc.binary_relations) == 6

        relation_tuples = [
            (str(rel.head), rel.label, str(rel.tail)) for rel in doc.binary_relations
        ]
        assert relation_tuples == [
            ("nicht mehr nehme", "SIGNALS_CHANGE_OF", "Citalopram"),
            ("Citalopram", "CAUSED", "massive Nervosität"),
            ("massive Nervosität", "CAUSED", "Brennen"),
            ("Citalopram", "CAUSED", "schreckliche Überdrehtheit"),
            ("gehts mir wieder viel besser", "IS_OPINION_ABOUT", "Citalopram"),
            ("dachte echt ich werde verrückt", "IS_OPINION_ABOUT", "schreckliche Überdrehtheit"),
        ]
    else:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def tokenized_documents_with_labeled_spans_and_binary_relations(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer
) -> Optional[List[TestTokenDocumentWithLabeledSpansAndBinaryRelations]]:
    if dataset_of_text_documents_with_labeled_spans_and_binary_relations is None:
        return None

    # get a document to check
    doc = dataset_of_text_documents_with_labeled_spans_and_binary_relations["train"][0]
    # Note, that this is a list of documents, because the document may be split into chunks
    # if the input text is too long.
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        return_overflowing_tokens=True,
        result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
        strict_span_conversion=False,
        verbose=True,
    )
    return tokenized_docs


def test_tokenized_documents_with_labeled_spans_and_binary_relations(
    tokenized_documents_with_labeled_spans_and_binary_relations,
):
    if tokenized_documents_with_labeled_spans_and_binary_relations is not None:
        docs = tokenized_documents_with_labeled_spans_and_binary_relations
        # check that the tokenization was fine
        assert len(docs) == 1
        doc = docs[0]
        assert len(doc.tokens) == 362
        # Check the first ten tokens
        assert doc.tokens[:10] == (
            "[CLS]",
            "hall",
            "##o",
            "lie",
            "##be",
            "<",
            "user",
            ">",
            ",",
            "dan",
        )
        assert len(doc.labeled_spans) == 10
        sorted_entity_tuples = [
            (str(ent), ent.label) for ent in sorted(doc.labeled_spans, key=lambda ent: ent.start)
        ]
        assert sorted_entity_tuples == [
            ("('br', '##enne', '##n')", "DISORDER"),
            ("('massive', 'ne', '##r', '##vos', '##ita', '##t')", "DISORDER"),
            ("('ci', '##tal', '##op', '##ram')", "DRUG"),
            ("('ang', '##st', '##patient', '##en')", "DISORDER"),
            ("('ad',)", "DRUG"),
            ("('ci', '##tal', '##op', '##ram')", "DRUG"),
            ("('nic', '##ht', 'me', '##hr', 'ne', '##hm', '##e')", "CHANGE_TRIGGER"),
            (
                "('ge', '##ht', '##s', 'mir', 'wi', '##ede', '##r', 'vie', '##l', 'be', '##sser')",
                "OPINION",
            ),
            (
                "('sc', '##hre', '##ck', '##liche', 'uber', '##dre', '##ht', '##hei', '##t')",
                "DISORDER",
            ),
            (
                "('da', '##cht', '##e', 'ec', '##ht', 'ich', 'we', '##rde', 've', '##rr', '##uck', '##t')",
                "OPINION",
            ),
        ]


def test_tokenized_documents_with_entities_and_relations_all(
    dataset_of_text_documents_with_labeled_spans_and_binary_relations, tokenizer, dataset_variant
):
    if dataset_of_text_documents_with_labeled_spans_and_binary_relations is not None:
        for (
            split,
            docs,
        ) in dataset_of_text_documents_with_labeled_spans_and_binary_relations.items():
            for doc in docs:
                # Note, that this is a list of documents, because the document may be split into chunks
                # if the input text is too long.
                tokenized_docs = tokenize_document(
                    doc,
                    tokenizer=tokenizer,
                    return_overflowing_tokens=True,
                    result_document_type=TestTokenDocumentWithLabeledSpansAndBinaryRelations,
                    strict_span_conversion=False,
                    verbose=True,
                )
                # we just ensure that we get at least one tokenized document
                assert tokenized_docs is not None
                assert len(tokenized_docs) > 0


def test_document_converters(dataset_variant):
    builder = Lisa(config_name=dataset_variant)
    document_converters = builder.document_converters
    if dataset_variant not in ["default", "merge_fragmented_spans"]:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")

    assert len(document_converters) == 1
    assert set(document_converters) == {
        TextDocumentWithLabeledSpansAndBinaryRelations,
    }
    assert all(callable(v) or isinstance(v, dict) for k, v in document_converters.items())

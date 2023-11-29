import os.path

from pie_datasets.builders import BratBuilder
from pie_datasets.core.dataset import DocumentConvertersType
from pytorch_ie.documents import TextDocumentWithLabeledSpansAndBinaryRelations

URL = os.path.abspath(
    "data/merged_2001_3000_positives_curated_deidentified_cleaned_format_fixed.zip"
)
SPLIT_PATHS = {"train": "merged_2001_3000_positives_curated_deidentified_cleaned_format_fixed"}


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
            return {}
        elif self.config.name == "merge_fragmented_spans":
            return {
                TextDocumentWithLabeledSpansAndBinaryRelations: {
                    "spans": "labeled_spans",
                    "relations": "binary_relations",
                }
            }
        else:
            raise ValueError(f"Unknown dataset variant: {self.config.name}")

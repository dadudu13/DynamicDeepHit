"""Where the data lives and what its columns are called.

Nothing in the evaluation scripts names a file or a column directly. Every path
comes from a command-line flag or, when the flag is omitted, an environment
variable; every column name comes from a Columns mapping, which defaults to the
generic names below and can be replaced with a JSON file (--columns or
DDH_COLUMNS). Patient-level data therefore stays wherever its owner keeps it.

Layout under the data root, one folder per random split:

    <data-root>/split_<k>/<template with part>      e.g. split_0/train.csv
        part = train, val, test                      one row per (patient, visit)
               train_landmark, test_landmark         fixed-landmark tables (data.build_landmark_table)
    <data-root>/split_<k>/<external template>       optional external / temporal cohort,
                                                     preprocessed with split k's transforms

Layout under the runs root, one folder per trained arm:

    <runs-root>/<arm>/split_<k>/                     model checkpoint, metadata.json, predictions
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENV_DATA_ROOT = "DDH_DATA_ROOT"
ENV_RUNS_ROOT = "DDH_RUNS_ROOT"
ENV_OUTPUT_DIR = "DDH_OUTPUT_DIR"
ENV_COLUMNS = "DDH_COLUMNS"
ENV_FILE_TEMPLATE = "DDH_FILE_TEMPLATE"
ENV_EXTERNAL_TEMPLATE = "DDH_EXTERNAL_TEMPLATE"
ENV_SPLIT_DIR_TEMPLATE = "DDH_SPLIT_DIR_TEMPLATE"

# Per-split files a training run writes and the evaluation reads.
FIXED_LANDMARK_FILE = "{part}_fixed_landmark_predictions.csv"
EVERY_VISIT_FILE = "{part}_every_visit_predictions.csv"
LAST_VISIT_FILE = "{part}_last_visit_predictions.csv"
EXTERNAL_EVERY_VISIT_FILE = "external_every_visit_predictions.csv"
EXTERNAL_ANCHOR_FILE = "external_anchor_predictions.csv"
CHECKPOINT_FILE = "model_state.pt"
METADATA_FILE = "metadata.json"

EVENT_OF_INTEREST = 1   # event column: 0 censored, 1 event of interest, 2 competing event
COMPETING_EVENT = 2


@dataclass
class Columns:
    """Names of the columns the pipeline needs. Everything else is a predictor."""

    id: str = "patient_id"
    date: str = "visit_date"
    visit: str = "visit"
    time: str = "time_to_event"        # days from the patient's FIRST visit to event or censoring
    event: str = "event"               # 0 censored, 1 event of interest, 2 competing event
    event_date: str = "event_date"     # optional; derived from date + time when absent
    categorical: str | None = None     # optional integer-coded column given a learned embedding
    meta: list[str] = field(default_factory=list)  # other columns that are not predictors

    def non_predictors(self) -> set[str]:
        landmark = {"landmark_day", "is_landmark", "is_prediction_row", "_gap_to_landmark",
                    "_date_orig"}
        return {self.id, self.date, self.visit, self.time, self.event, self.event_date,
                *self.meta, *landmark}

    @classmethod
    def load(cls, path: str | Path | None) -> "Columns":
        path = path or os.environ.get(ENV_COLUMNS)
        if not path:
            return cls()
        values = json.loads(Path(path).read_text())
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown column keys {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Paths:
    data_root: Path | None
    runs_root: Path | None
    output_dir: Path | None
    file_template: str = "{part}.csv"
    external_template: str = "external.csv"
    split_dir_template: str = "split_{split}"

    def split_file(self, split: int, part: str) -> Path:
        return self._need(self.data_root, "--data-root", ENV_DATA_ROOT) / \
            f"split_{split}" / self.file_template.format(part=part)

    def external_file(self, split: int) -> Path:
        return self._need(self.data_root, "--data-root", ENV_DATA_ROOT) / \
            f"split_{split}" / self.external_template.format(split=split)

    def run_dir(self, arm: str, split: int) -> Path:
        arm_root = Path(arm)
        if not arm_root.is_absolute():
            arm_root = self._need(self.runs_root, "--runs-root", ENV_RUNS_ROOT) / arm
        return arm_root / self.split_dir_template.format(split=split)

    def out(self, name: str) -> Path:
        out = self._need(self.output_dir, "--out", ENV_OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        return out / name

    @staticmethod
    def _need(value: Path | None, flag: str, env: str) -> Path:
        if value is None:
            raise SystemExit(f"Set {flag} or the {env} environment variable.")
        return value


def add_path_arguments(parser: argparse.ArgumentParser, data=True, runs=True, out=True) -> None:
    """The flags every script shares. Each falls back to its environment variable."""
    if data:
        parser.add_argument("--data-root", default=os.environ.get(ENV_DATA_ROOT),
                            help=f"Folder holding split_<k>/ (env {ENV_DATA_ROOT}).")
        parser.add_argument("--file-template", default=os.environ.get(ENV_FILE_TEMPLATE, "{part}.csv"),
                            help="File name inside split_<k>/, with {part} for train, val, test, "
                                 f"train_landmark, test_landmark (env {ENV_FILE_TEMPLATE}; "
                                 "default '{part}.csv').")
        parser.add_argument("--external-template",
                            default=os.environ.get(ENV_EXTERNAL_TEMPLATE, "external.csv"),
                            help="External cohort file inside split_<k>/; may use {split} "
                                 f"(env {ENV_EXTERNAL_TEMPLATE}; default 'external.csv').")
    if runs:
        parser.add_argument("--runs-root", default=os.environ.get(ENV_RUNS_ROOT),
                            help=f"Folder holding one sub-folder per trained arm (env {ENV_RUNS_ROOT}).")
        parser.add_argument("--split-dir-template",
                            default=os.environ.get(ENV_SPLIT_DIR_TEMPLATE, "split_{split}"),
                            help="Name of a split's folder inside an arm "
                                 f"(env {ENV_SPLIT_DIR_TEMPLATE}; default 'split_{{split}}').")
    if out:
        parser.add_argument("--out", default=os.environ.get(ENV_OUTPUT_DIR),
                            help=f"Folder for result tables (env {ENV_OUTPUT_DIR}).")
    parser.add_argument("--columns", default=None,
                        help=f"JSON file mapping Columns fields to your column names (env {ENV_COLUMNS}).")


def paths_from_args(args: argparse.Namespace) -> Paths:
    def p(name):
        value = getattr(args, name, None)
        return Path(value).expanduser().resolve() if value else None

    return Paths(
        data_root=p("data_root"),
        runs_root=p("runs_root"),
        output_dir=p("out"),
        file_template=getattr(args, "file_template", "{part}.csv"),
        external_template=getattr(args, "external_template", "external.csv"),
        split_dir_template=getattr(args, "split_dir_template", "split_{split}"),
    )

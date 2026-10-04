"""Turn an attribute file into a data variant the training code reads.

data/codraft/ was built from data/term_attributes.csv by one rule, recovered from the files:
each side's Nature / Purpose is the attribute of its term, lower-cased, matched on the
lower-cased term. build_variant applies the same rule to any attribute file, so a variant
from another LLM differs from data/codraft/ in the attributes and nothing else: same rows,
same order, same Pair IDs, same labels. check_rule() rebuilds data/codraft/ from the Gemini
cache and asserts it comes out identical.
"""
import os

import pandas as pd

from .enrich import norm

SPLITS = ("train", "val", "test")


def load_attributes(path):
    """{(term, class) or term: (nature, purpose)}. A file without a Class column (the Gemini
    cache) is keyed on the term alone, first occurrence kept, as data/codraft/ was."""
    a = pd.read_csv(path, low_memory=False, keep_default_na=False)
    by_class = "Class" in a.columns
    out = {}
    for r in a.itertuples(index=False):
        k = (norm(r.Original_Term), int(r.Class)) if by_class else norm(r.Original_Term)
        out.setdefault(k, (str(r.Nature), str(r.Purpose)))
    return out, by_class


def build_variant(data_root, attributes_csv, name, out_root=None, splits=None, pair_ids=None):
    """Write <out_root>/<name>/{split}.csv from data_root/codraft/ with the new attributes.

    splits: which splits to write; by default every split the attributes fully cover (so a
    test-only enrichment gives a test-only variant). A split asked for by name must be fully
    covered. Returns {split: number of rows}.
    pair_ids: keep only these pairs, in their data/codraft/ order (the smoke test's subset).
    """
    attrs, by_class = load_attributes(attributes_csv)
    out_dir = os.path.join(out_root or data_root, name)
    written, skipped = {}, {}
    for split in (splits or SPLITS):
        base = pd.read_csv(os.path.join(data_root, "codraft", f"{split}.csv"), low_memory=False)
        if pair_ids is not None:
            base = base[base["Pair ID"].isin(set(pair_ids))].reset_index(drop=True)
        d = base.copy()
        ok = True
        for side in ("1", "2"):
            keys = [(norm(t), int(c)) if by_class else norm(t)
                    for t, c in zip(d[f"Term {side}"], d[f"Class {side}"])]
            hit = [k in attrs for k in keys]
            if not all(hit):
                ok = False
                skipped[split] = skipped.get(split, 0) + hit.count(False)
                continue
            d[f"Nature {side}"] = [attrs[k][0].lower() for k in keys]
            d[f"Purpose {side}"] = [attrs[k][1].lower() for k in keys]
        if not ok:
            if splits:
                raise ValueError(f"{split}: {skipped[split]} term sides have no attributes in "
                                 f"{attributes_csv}")
            continue
        for c in ("Nature 1", "Purpose 1", "Nature 2", "Purpose 2"):
            empty = (d[c].str.strip() == "").sum()
            assert empty == 0, f"{split}: {empty} empty values in {c}"
        assert list(d["Pair ID"]) == list(base["Pair ID"])
        assert list(d["Similarity"]) == list(base["Similarity"])
        os.makedirs(out_dir, exist_ok=True)
        d.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        written[split] = len(d)
    if not written:
        raise ValueError(f"the attributes cover no split completely (missing term sides: {skipped})")
    print(f"variant {name}: wrote {written} to {out_dir}"
          + (f"; not covered: {skipped}" if skipped else ""))
    return written


def check_rule(data_root, tmp_dir):
    """Rebuild data/codraft/ from term_attributes.csv; raises unless identical."""
    build_variant(data_root, os.path.join(data_root, "term_attributes.csv"), "codraft_rebuilt",
                  out_root=tmp_dir, splits=SPLITS)
    for split in SPLITS:
        a = pd.read_csv(os.path.join(data_root, "codraft", f"{split}.csv"), low_memory=False)
        b = pd.read_csv(os.path.join(tmp_dir, "codraft_rebuilt", f"{split}.csv"), low_memory=False)
        pd.testing.assert_frame_equal(a, b)
    print("check: term_attributes.csv rebuilds data/codraft/ exactly")
    return True

import os

import pandas as pd

from codraft_enrichment.enrich import norm

SPLITS = ("train", "val", "test")


def load_attributes(path: str) -> tuple[dict, bool]:
    a = pd.read_csv(path, low_memory=False, keep_default_na=False)
    by_class = "Class" in a.columns
    out = {}
    for r in a.itertuples(index=False):
        k = (norm(r.Original_Term), int(r.Class)) if by_class else norm(r.Original_Term)
        out.setdefault(k, (str(r.Nature), str(r.Purpose)))
    return out, by_class


def build_variant(
    data_root: str,
    attributes_csv: str,
    name: str,
    out_root: str | None = None,
    splits: tuple[str, ...] | None = None,
    pair_ids: list[str] | None = None,
) -> dict:
    attrs, by_class = load_attributes(attributes_csv)
    out_dir = os.path.join(out_root or data_root, name)
    written, skipped = {}, {}
    for split in splits or SPLITS:
        base = pd.read_csv(os.path.join(data_root, "codraft", f"{split}.csv"), low_memory=False)
        if pair_ids is not None:
            base = base[base["Pair ID"].isin(set(pair_ids))].reset_index(drop=True)
        d = base.copy()
        missing = 0
        for side in ("1", "2"):
            keys = [
                (norm(t), int(c)) if by_class else norm(t)
                for t, c in zip(d[f"Term {side}"], d[f"Class {side}"])
            ]
            missing += sum(k not in attrs for k in keys)
            if not missing:
                d[f"Nature {side}"] = [attrs[k][0].lower() for k in keys]
                d[f"Purpose {side}"] = [attrs[k][1].lower() for k in keys]
        if missing:
            skipped[split] = missing
            if splits:
                raise ValueError(
                    f"{split}: {missing} term sides have no attributes in {attributes_csv}"
                )
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
        raise ValueError(
            f"The attributes cover no split completely (missing term sides: {skipped})"
        )
    print(
        f"Variant {name}: wrote {written} to {out_dir}"
        + (f"; not covered: {skipped}" if skipped else "")
    )
    return written


def check_rule(data_root: str, tmp_dir: str) -> None:
    attributes = os.path.join(data_root, "term_attributes.csv")
    build_variant(data_root, attributes, "codraft_rebuilt", out_root=tmp_dir, splits=SPLITS)
    for split in SPLITS:
        a = pd.read_csv(os.path.join(data_root, "codraft", f"{split}.csv"), low_memory=False)
        b = pd.read_csv(os.path.join(tmp_dir, "codraft_rebuilt", f"{split}.csv"), low_memory=False)
        pd.testing.assert_frame_equal(a, b)
    print("Check: term_attributes.csv rebuilds data/codraft/ exactly")

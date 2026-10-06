import pandas as pd

from config.config_data import CONFIG_DATA

LABEL_MAPPING = {"Dissimilar": 0, "Low similar": 1, "Similar": 2, "High similar": 3, "Identical": 4}


def _given(value: object) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def create_structured_text_enhanced(
    term: str, nature: object, purpose: object, class_id: int
) -> str:
    parts = []
    if _given(nature):
        parts.append(f"Nature: {str(nature).strip()}")
    if _given(purpose):
        parts.append(f"Use: {str(purpose).strip()}")
    heading = CONFIG_DATA.NICE_CLASS_MAP.get(int(class_id), "")
    if heading:
        parts.append(f"Category: {heading}")
    text = str(term).strip()
    return f"{text} [ {' | '.join(parts)} ]" if parts else text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["label_score"] = df["Similarity"].map(LABEL_MAPPING)
    df = df.dropna(subset=["label_score"])
    df["label_score"] = df["label_score"].astype(int)
    for side in ("1", "2"):
        if f"Term {side} Expand" in df.columns:
            df[f"input_text_{side}"] = df[f"Term {side} Expand"].astype(str).str.strip()
        elif f"Nature {side}" in df.columns:
            df[f"input_text_{side}"] = [
                create_structured_text_enhanced(t, n, p, c)
                for t, n, p, c in zip(
                    df[f"Term {side}"],
                    df[f"Nature {side}"],
                    df[f"Purpose {side}"],
                    df[f"Class {side}"],
                )
            ]
        else:
            df[f"input_text_{side}"] = df[f"Term {side}"].astype(str).str.strip()
    return df

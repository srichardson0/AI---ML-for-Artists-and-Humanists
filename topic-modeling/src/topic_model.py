"""Train and evaluate an LDA model for CUNY Graduate Center abstracts."""

from pathlib import Path
import re

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from custom_stop_words_pattern import my_stop_words


DO_COHERENCE_TEST = False
START = 10
LIMIT = 100
STEP = 15
NUM_TOPICS = 25
PASSES = 10
WORKERS = 8
RANDOM_STATE = 2026
USE_SUBSET = False
TRUNCATE_SIZE = 50000
VERBOSE = True
MIN_WORDS = 15
MAX_WORDS = 100000
MAX_PERCENTAGE = 0.5
DROP_DUPLICATES = True

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
MODEL_DIR = PROJECT_DIR / "models" / "lda"
SCORES_DIR = PROJECT_DIR / "outputs" / "scores"
EVALUATION_DIR = PROJECT_DIR / "outputs" / "evaluation"
CSV_FILE = DATA_DIR / "gc_dissertations_combined_v2.csv"
MODEL_FILE = MODEL_DIR / "lda_model.gensim"
DICTIONARY_FILE = MODEL_DIR / "lda_model.dict"
SCORES_FILE = SCORES_DIR / "dissertation_topic_scores.csv"
COHERENCE_FILE = EVALUATION_DIR / "coherence_values.csv"
COHERENCE_PLOT = EVALUATION_DIR / "coherence_values.png"
TEXT_COLUMN = "abstract"

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def lemmatize_stemming(text):
    return stemmer.stem(lemmatizer.lemmatize(text, pos="v"))


def preprocess(text):
    """Apply the same token transformation during training and inference."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    return [
        lemmatize_stemming(token)
        for token in gensim.utils.simple_preprocess(text)
        if token not in my_stop_words and len(token) > 3
    ]


def discipline_from_program(program):
    """Map program labels to Humanities, Social Sciences, Sciences, or Mixed."""
    value = re.sub(r"[^a-z]+", " ", str(program).lower()).strip()
    if not value or value in {"not found", "wrong link"}:
        return "Mixed/Uncertain"
    humanities = ("art history", "fine art", "decorative art", "architecture", "classics", "comparative literature", "cinema", "design", "english", "french", "german", "history", "jewish", "language", "latin", "literature", "middle eastern", "music", "philosoph", "portuguese", "religion", "spanish", "theater", "theatre", "women", "gender", "american studies", "digital humanities")
    sciences = ("agriculture", "astrophysics", "biochem", "biology", "biophys", "chem", "computer science", "cognitive neuroscience", "earth", "engineering", "geochem", "geolog", "geophys", "health science", "hydrolog", "immunolog", "mathematics", "mechanical", "nano", "neuro", "paleo", "pharmac", "pharmacy", "physics", "physiology", "remote sensing", "statistics", "speech", "audiology", "nursing", "physical therapy", "hearing", "environmental science")
    social_sciences = ("account", "anthropolog", "business", "buisness", "criminal justice", "demograph", "econom", "education", "gerontolog", "geography", "international migration", "international politic", "law", "mass communication", "political", "public administration", "public health", "recreation", "social", "sociolog", "psycholog", "urban", "transportation", "liberal studies")
    matches = [any(term in value for term in group) for group in (humanities, social_sciences, sciences)]
    if sum(matches) != 1:
        return "Mixed/Uncertain"
    return ("Humanities", "Social Sciences", "Sciences")[matches.index(True)]


def load_documents():
    data = pd.read_csv(CSV_FILE)
    required = {"record_id", "program_coarse", TEXT_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["macro_discipline"] = data["program_coarse"].map(discipline_from_program)
    valid = data[TEXT_COLUMN].notna() & data[TEXT_COLUMN].map(lambda value: isinstance(value, str))
    if "abstract_is_placeholder" in data:
        valid &= data["abstract_is_placeholder"].fillna(0).eq(0)
    data = data.loc[valid].copy()
    if DROP_DUPLICATES and "dup_group" in data:
        data = data.loc[data["dup_group"].isna() | ~data.duplicated("dup_group")].copy()
    if USE_SUBSET:
        data = data.head(TRUNCATE_SIZE).copy()
    data["tokens"] = data[TEXT_COLUMN].map(preprocess)
    data = data.loc[data["tokens"].map(bool)].reset_index(drop=True)
    if VERBOSE:
        print(f"Eligible documents: {len(data)}")
        print(data["macro_discipline"].value_counts().to_string())
    return data


def compute_coherence_values(dictionary, corpus, texts):
    topic_counts = list(range(START, LIMIT + 1, STEP))
    coherence_values = []
    for num_topics in topic_counts:
        model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=PASSES, workers=WORKERS, random_state=RANDOM_STATE)
        score = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()
        coherence_values.append(score)
        print(f"{num_topics} topics: coherence={score:.4f}")
    return topic_counts, coherence_values


def score_documents(model, dictionary, data):
    rows = []
    for _, row in data.iterrows():
        distribution = model.get_document_topics(dictionary.doc2bow(row["tokens"]), minimum_probability=0)
        scores = np.array([score for _, score in distribution], dtype=float)
        result = row.drop(labels="tokens").to_dict()
        result["dominant_topic"] = int(scores.argmax())
        result["topic_entropy"] = float(-(scores[scores > 0] * np.log(scores[scores > 0])).sum())
        result.update({f"topic_{index}": float(score) for index, score in enumerate(scores)})
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    nltk.download("wordnet", quiet=True)
    np.random.seed(RANDOM_STATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    data = load_documents()
    texts = data["tokens"].tolist()
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=MIN_WORDS, no_above=MAX_PERCENTAGE, keep_n=MAX_WORDS)
    corpus = [dictionary.doc2bow(text) for text in texts]
    nonempty = [index for index, bow in enumerate(corpus) if bow]
    if len(nonempty) != len(corpus):
        data = data.iloc[nonempty].reset_index(drop=True)
        corpus = [corpus[index] for index in nonempty]
    if DO_COHERENCE_TEST:
        topic_counts, coherence_values = compute_coherence_values(dictionary, corpus, data["tokens"].tolist())
        pd.DataFrame({"num_topics": topic_counts, "coherence": coherence_values}).to_csv(COHERENCE_FILE, index=False)
        plt.plot(topic_counts, coherence_values, marker="o")
        plt.xlabel("Number of topics")
        plt.ylabel("Coherence score")
        plt.title("Topic coherence by number of topics")
        plt.savefig(COHERENCE_PLOT, dpi=150, bbox_inches="tight")
        return
    model = gensim.models.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=PASSES, workers=WORKERS, random_state=RANDOM_STATE)
    for index, topic in model.print_topics(-1):
        print(f"Topic {index}: {topic}")
    model.save(str(MODEL_FILE))
    dictionary.save(str(DICTIONARY_FILE))
    scored = score_documents(model, dictionary, data)
    scored.to_csv(SCORES_FILE, index=False)
    print(f"Saved model, dictionary, and {len(scored)} scored documents.")


if __name__ == "__main__":
    main()

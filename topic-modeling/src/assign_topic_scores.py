"""Apply a saved topic model using the training pipeline's preprocessing."""

from topic_model import (
    DICTIONARY_FILE,
    MODEL_FILE,
    SCORES_FILE,
    load_documents,
    score_documents,
)
import gensim


REFRESHED_SCORES_FILE = SCORES_FILE.with_name("dissertation_topic_scores_refreshed.csv")


if __name__ == "__main__":
    model = gensim.models.LdaMulticore.load(str(MODEL_FILE))
    dictionary = gensim.corpora.Dictionary.load(str(DICTIONARY_FILE))
    data = load_documents()
    scored = score_documents(model, dictionary, data)
    scored.to_csv(REFRESHED_SCORES_FILE, index=False)
    print(f"Saved topic scores for {len(scored)} documents to {REFRESHED_SCORES_FILE}.")

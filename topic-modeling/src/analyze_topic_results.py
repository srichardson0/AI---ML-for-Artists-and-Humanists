"""Summarize the saved LDA results by discipline and create figures."""

from pathlib import Path

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCORES_FILE = PROJECT_DIR / "outputs" / "scores" / "dissertation_topic_scores_refreshed.csv"
MODEL_FILE = PROJECT_DIR / "models" / "lda" / "lda_model.gensim"
COHERENCE_FILE = PROJECT_DIR / "outputs" / "evaluation" / "coherence_values.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "analysis"
SHARED_TOPIC_THRESHOLD = 0.02
TOP_N = 10
DISCIPLINE_ORDER = ["Humanities", "Social Sciences", "Sciences", "Mixed/Uncertain"]


def load_results():
    data = pd.read_csv(SCORES_FILE, low_memory=False)
    topic_columns = sorted(
        [column for column in data.columns if column.startswith("topic_") and column != "topic_entropy"],
        key=lambda column: int(column.split("_")[1]),
    )
    data[topic_columns] = data[topic_columns].apply(pd.to_numeric, errors="coerce")
    data["topic_entropy"] = pd.to_numeric(data["topic_entropy"], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data.loc[data["macro_discipline"].isin(DISCIPLINE_ORDER)].copy()
    return data, topic_columns


def topic_label(model, topic_id):
    words = [word for word, _ in model.show_topic(topic_id, topn=8)]
    return ", ".join(words)


def save_tables(data, topic_columns, model):
    OUTPUT_DIR.mkdir(exist_ok=True)
    counts = data["macro_discipline"].value_counts().reindex(DISCIPLINE_ORDER, fill_value=0)
    pd.DataFrame({"macro_discipline": counts.index, "documents": counts.values, "percent": counts.values / len(data) * 100}).to_csv(OUTPUT_DIR / "discipline_counts.csv", index=False)

    means = data.groupby("macro_discipline")[topic_columns].mean().reindex(DISCIPLINE_ORDER)
    means.to_csv(OUTPUT_DIR / "discipline_topic_means.csv")
    overall = data[topic_columns].mean()
    labels = pd.DataFrame({
        "topic_id": [int(column.split("_")[1]) for column in topic_columns],
        "top_words": [topic_label(model, int(column.split("_")[1])) for column in topic_columns],
        "overall_mean_proportion": overall.values,
    })
    labels.to_csv(OUTPUT_DIR / "topic_labels.csv", index=False)

    distinctive_rows = []
    for discipline in DISCIPLINE_ORDER[:3]:
        contrast = means.loc[discipline] - means.drop(index=discipline).mean()
        for column in contrast.nlargest(TOP_N).index:
            topic_id = int(column.split("_")[1])
            distinctive_rows.append({
                "macro_discipline": discipline,
                "topic_id": topic_id,
                "mean_proportion": means.loc[discipline, column],
                "contrast_vs_other_disciplines": contrast[column],
                "top_words": labels.loc[labels["topic_id"] == topic_id, "top_words"].iloc[0],
            })
    pd.DataFrame(distinctive_rows).to_csv(OUTPUT_DIR / "distinctive_topics.csv", index=False)

    shared_rows = []
    for column in topic_columns:
        present = means.loc[DISCIPLINE_ORDER[:3], column] >= SHARED_TOPIC_THRESHOLD
        if present.sum() >= 2:
            topic_id = int(column.split("_")[1])
            shared_rows.append({
                "topic_id": topic_id,
                "disciplines_at_threshold": "; ".join(present.index[present]),
                "number_of_disciplines": int(present.sum()),
                "overall_mean_proportion": overall[column],
                "top_words": labels.loc[labels["topic_id"] == topic_id, "top_words"].iloc[0],
            })
    pd.DataFrame(shared_rows).sort_values("overall_mean_proportion", ascending=False).to_csv(OUTPUT_DIR / "cross_disciplinary_topics.csv", index=False)
    return counts, means, labels


def save_figures(data, topic_columns, counts, means, labels):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, color=["#9b5de5", "#f15bb5", "#00bbf9", "#777777"])
    ax.set_ylabel("Eligible dissertations")
    ax.set_xlabel("")
    ax.set_title("Dissertation abstracts by discipline")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "discipline_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5))
    image = ax.imshow(means.loc[DISCIPLINE_ORDER[:3]].values, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(3), DISCIPLINE_ORDER[:3])
    ax.set_xticks(range(len(topic_columns)), [column.split("_")[1] for column in topic_columns], rotation=90)
    ax.set_xlabel("Topic ID")
    ax.set_title("Mean topic proportion by discipline")
    fig.colorbar(image, ax=ax, label="Mean topic proportion")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "discipline_topic_heatmap.png", dpi=180)
    plt.close(fig)

    top_topics = labels.nlargest(12, "overall_mean_proportion")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top_topics["topic_id"].astype(str), top_topics["overall_mean_proportion"], color="#277da1")
    ax.invert_yaxis()
    ax.set_xlabel("Mean topic proportion")
    ax.set_ylabel("Topic ID")
    ax.set_title("Most prevalent topics across the corpus")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overall_topic_prevalence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = [data.loc[data["macro_discipline"] == discipline, "topic_entropy"].dropna() for discipline in DISCIPLINE_ORDER[:3]]
    ax.boxplot(grouped, tick_labels=DISCIPLINE_ORDER[:3], showfliers=False)
    ax.set_ylabel("Document-topic entropy")
    ax.set_title("Topic-distribution breadth by discipline")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "topic_entropy_by_discipline.png", dpi=180)
    plt.close(fig)


def save_report(data, topic_columns, counts, means, labels):
    coherence = pd.read_csv(COHERENCE_FILE)
    best = coherence.loc[coherence["coherence"].idxmax()]
    top_topics = labels.nlargest(8, "overall_mean_proportion")
    shared = pd.read_csv(OUTPUT_DIR / "cross_disciplinary_topics.csv")
    lines = [
        "# Topic Modeling Findings",
        "",
        "## Research Question",
        "",
        "> How are research themes distributed across the humanities, social sciences, and sciences in CUNY Graduate Center dissertations, and which themes cross disciplinary boundaries?",
        "",
        "## Corpus and Model",
        "",
        f"The analysis includes **{len(data):,} dissertation abstracts** after excluding missing, placeholder, empty, and duplicate records. The discipline mapping assigns programs to Humanities, Social Sciences, Sciences, or Mixed/Uncertain. The primary comparisons below use the first three groups; Mixed/Uncertain remains visible in the corpus counts.",
        "",
        f"The coherence sweep favored a **{int(best['num_topics'])}-topic model** (coherence = **{best['coherence']:.4f}**). The nearby 70-topic model was similar, so the topic count is treated as a practical interpretive choice rather than a uniquely correct answer.",
        "",
        "## Corpus Distribution",
        "",
    ]
    for discipline in DISCIPLINE_ORDER:
        lines.append(f"- **{discipline}:** {int(counts[discipline]):,} dissertations ({counts[discipline] / len(data) * 100:.1f}%)")
    lines += ["", "## Most Prevalent Topics", "", "The most prevalent topics in the corpus were:", ""]
    for _, row in top_topics.iterrows():
        lines.append(f"- **Topic {int(row['topic_id'])}** ({row['overall_mean_proportion']:.3f}): {row['top_words']}")
    lines += ["", "## Cross-Disciplinary Themes", "", f"A topic is marked as potentially cross-disciplinary when its mean proportion is at least {SHARED_TOPIC_THRESHOLD:.2f} in at least two of the three primary discipline groups. This is an operational definition of overlap, not proof that the groups investigate the same questions.", ""]
    if shared.empty:
        lines.append("No topics met the current threshold in at least two groups.")
    else:
        for _, row in shared.head(12).iterrows():
            lines.append(f"- **Topic {int(row['topic_id'])}** ({row['disciplines_at_threshold']}): {row['top_words']}")
    lines += ["", "## Interpretation", "", "The heatmap and distinctive-topic table should be used together. A high topic proportion in one discipline indicates concentration, while a topic that remains substantial across multiple disciplines is a candidate cross-boundary theme. Topic labels are analyst summaries of word distributions and should be checked against representative dissertation abstracts.", "", "## Limitations", "", "- LDA models recurring word co-occurrence, not research themes as fixed or objective categories.", "- The discipline groups are derived from program metadata and include subjective mapping decisions.", "- Abstract availability and duplicate removal may change the apparent distribution of themes.", "- Stemming makes topic words compact but sometimes less readable.", "- Cross-disciplinary status depends on the chosen prevalence threshold and should be tested for robustness.", ""]
    (OUTPUT_DIR / "findings_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    data, topic_columns = load_results()
    model = gensim.models.LdaModel.load(str(MODEL_FILE))
    counts, means, labels = save_tables(data, topic_columns, model)
    save_figures(data, topic_columns, counts, means, labels)
    save_report(data, topic_columns, counts, means, labels)
    print(f"Analyzed {len(data):,} scored documents and wrote results to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()

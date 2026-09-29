

Interdisciplinarity! Are certain programs more interdisciplinary than others? Are there overriding topics/themes that pervade the GC? Are there meta topics? Does the topic model match up with GC departments – which is to say is this telling us anything more than the “program” column already is?


Research Question:

How do research themes differ between the humanities, social sciences, and sciences?

## Project Layout

- `src/`: training, scoring, analysis, and preprocessing scripts.
- `notebooks/`: exploratory and Colab notebooks.
- `data/raw/`: canonical dissertation CSV; the backup copy is in `data/raw/archive/`.
- `models/lda/`: saved Gensim model and dictionary artifacts.
- `outputs/scores/`: document-topic score exports.
- `outputs/evaluation/`: coherence tables and plots.
- `outputs/analysis/`: summary tables, figures, and findings report.

Run the local pipeline from the workspace root:

```powershell
python ".\Topic Modeling\src\topic_model.py"
python ".\Topic Modeling\src\assign_topic_scores.py"
python ".\Topic Modeling\src\analyze_topic_results.py"
```

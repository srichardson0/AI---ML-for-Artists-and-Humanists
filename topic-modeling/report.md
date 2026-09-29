**Sasha Richardson**

**DATA 78000: AI and Machine Learning for Artists and Humanists, CUNY Graduate Center**

**Topic Modeling the Graduate Center's Intellectual History**

## 1. Introduction and Research Question

The CUNY Graduate Center's dissertation and thesis archive spans decades of work across the humanities, social sciences, and sciences, but no single scholar could read it closely enough to describe its overall shape. Topic modeling offers a way to approximate that reading: by tracking which words recur together across thousands of abstracts, it is possible to surface latent themes without processing each document by hand.

This assignment uses Latent Dirichlet Allocation (LDA) to ask:

**How do research themes differ between the humanities, social sciences, and sciences at the Graduate Center?**



## 2. Data and Preprocessing

### 2.1 Corpus

The source data is a CSV of Graduate Center dissertation and thesis records, including an abstract field and a program label for each record. The raw file contains roughly 18,800 records.

Records were excluded from the analysis if their abstract was missing, flagged as a placeholder, or part of a detected duplicate group. After this filtering, the working corpus contained **16,301 dissertation abstracts**.

Each record's academic program was mapped to one of four discipline groups — **Humanities**, **Social Sciences**, **Sciences**, or **Mixed/Uncertain** — using a keyword match against the program name (e.g., "history," "philosophy," and "art history" map to Humanities; "sociology," "economics," and "psychology" map to Social Sciences; "physics," "biology," and "computer science" map to Sciences). Programs that matched more than one group, or none, were labeled Mixed/Uncertain. This mapping is subjective and not ground truth.

| Discipline | Dissertations | Share |
|---|---|---|
| Social Sciences | 6,836 | 41.9% |
| Sciences | 4,918 | 30.2% |
| Humanities | 4,044 | 24.8% |
| Mixed/Uncertain | 503 | 3.1% |

The main comparisons in this report use the three primary groups (Humanities, Social Sciences, Sciences); Mixed/Uncertain is included in the corpus totals but not treated as a fourth analytical category.

![Discipline distribution](analysis/discipline_counts.png)

### 2.2 Text Preprocessing

Each abstract was tokenized and cleaned before modeling, using the following steps:

1. **Tokenization and normalization** with `gensim.utils.simple_preprocess`, which lowercases text and strips punctuation and numerals.
2. **Stop word removal**, using Gensim's default English stop word list extended with a small custom list of corpus-specific noise terms ('lcub', 'rcub', 'dollar', 'work', 'dissertation', 'thesis', 'examine', 'chapter', 'high', 'result').
3. **Length filtering**, dropping tokens of 3 characters or fewer.
4. **Lemmatization and stemming**, first lemmatizing each token as a verb form and then applying a Snowball (Porter-style) English stemmer. This compresses related word forms (e.g., "analyze," "analyzing," "analysis") into a single token, at the cost of producing truncated, sometimes less readable word stems in the topic output (e.g., "polit," "cultur," "signific").
5. **Dictionary filtering**, discarding tokens that appeared in fewer than 15 documents or in more than 50% of documents, which removes both idiosyncratic noise and terms so common they carry no discriminating power.
6. Documents that had no tokens left after this pipeline were dropped from the corpus.


## 3. Model Training
An LDA model was trained with Gensim across a range of topic counts (10 to 100, in steps of 15), scoring each with the `c_v` coherence metric, which estimates how semantically consistent each topic's top words are.

![Coherence by number of topics](evaluation/coherence_values.png)

| Number of topics | Coherence (c_v) |
|---|---|
| 10 | 0.466 |
| **25** | **0.485** |
| 40 | 0.462 |
| 55 | 0.462 |
| 70 | 0.459 |
| 85 | 0.440 |
| 100 | 0.436 |

Coherence peaked at **25 topics (0.485)** and declined gradually beyond that point. The final model was trained with 25 topics, 10 passes over the corpus, and a fixed random seed for reproducibility.

## 4. Results

### 4.1 Most Prevalent Topics Overall

Averaged across the whole corpus, the most prevalent topics were:

| Topic | Mean proportion | Top words |
|---|---|---|
| 15 | 0.104 | polit, american, cultur, state, nation, social, histori, black |
| 17 | 0.082 | artist, music, work, centuri, perform, play, cultur, text |
| 1 | 0.060 | group, self, measur, effect, signific, behavior, research, particip |
| 0 | 0.058 | women, narrat, novel, gender, write, social, literari, self |
| 12 | 0.050 | state, time, model, energi, optic, measur, effect, result |
| 8 | 0.048 | theori, argu, social, view, moral, natur, cultur, understand |
| 18 | 0.047 | children, group, word, task, differ, languag, result, effect |
| 10 | 0.046 | model, speci, data, result, differ, estim, analysi, base |

![Overall topic prevalence](analysis/overall_topic_prevalence.png)

The two largest topics in the whole corpus are historical/political-cultural themes (Topic 15) and arts/performance/humanities themes (Topic 17), consistent with a corpus where Social Sciences and Humanities together outnumber Sciences.

### 4.2 Topics by Discipline

The heatmap below shows the mean proportion of each topic within each discipline group, making it possible to see which topics concentrate in one discipline versus spreading across several.

![Discipline by topic heatmap](analysis/discipline_topic_heatmap.png)

The topics that most distinctly characterize each discipline (highest mean proportion relative to the other groups) were:

**Humanities:** Topic 17 (artist, music, work, centuri, perform, play, cultur, text), Topic 15 (polit, american, cultur, state, nation, social, histori, black), Topic 0 (women, narrat, novel, gender, write, social, literari, self), Topic 8 (theori, argu, social, view, moral, natur, cultur, understand).

**Social Sciences:** Topic 1 (group, self, measur, effect, signific, behavior, research, particip), Topic 4 (market, firm, effect, increas, manag, chang, financi, industri), Topic 16 (treatment, behavior, patient, famili, stress, associ, signific, effect), Topic 19 (citi, york, immigr, polici, state, public, social, communiti).

**Sciences:** Topic 12 (state, time, model, energi, optic, measur, effect, result), Topic 2 (surfac, group, structur, properti, materi, theori, function, field), Topic 3 (cell, protein, activ, gene, express, regul, induc, function), Topic 13 (receptor, structur, reaction, effect, synthesi, bind, activ, result).

These distinctive topics broadly track the intuitive character of each discipline group. Literary and historical themes for Humanities, institutional and behavioral themes for Social Sciences, and physical/biochemical themes for Sciences; which is a useful sanity check that the model is capturing something real, even though it was given no discipline information during training.


## 5. Interpretation

Read together, the heatmap and the distinctive/cross-disciplinary tables suggest a Graduate Center research landscape organized less around three cleanly separated disciplines and more around a few connective themes — cultural/political history, gender and narrative, and applied modeling/measurement — that recur under different guises across discipline boundaries. The Humanities–Social Sciences bridge is broader than the Sciences' connection to either group, which tracks with Sciences abstracts relying on a more specialized, technical vocabulary (protein, receptor, synthesis, optic) that has less natural overlap with the vocabulary of narrative, policy, or theory.




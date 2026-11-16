# Agentic RAG with evaluation over SEC 10-K filings: report

This report describes what was built, why each choice was made, how the systems were evaluated, and what the results do
and do not show. Every number in it comes from files in this repository (`results/`, `questions/`) and can be
recomputed with `python -m src.eval.grades_report`. Choices that were measured are labelled as such; choices that were
only reasoned (not tested against alternatives) are labelled as such too.

## 1. Summary

* **Task.** Answer questions about the annual reports (10-K) of 19 US companies, cite the passages used, and refuse
  when the filings do not contain the answer.
* **Systems compared** (same language model, same prompt): `rag_base` (dense search, 5 passages), `rag5` (company/year
  filter + reranker, 5 passages), `rag8` (filter + keyword/dense hybrid + reranker, 8 passages) and `agent_v3` (a
  LangGraph agent that plans one search per company and year, checks what is missing and searches again).
* **Evaluation.** 175 questions with hand-written answers and evidence passages; 94 of them (test and agentic sets)
  graded by hand, blind, against the reference answer.
* **Main result.** Fully correct answers on the 94 graded questions: 71.3% (`rag_base`), 74.5% (`rag5`), 80.9%
  (`rag8`), 83.0% (`agent_v3`). Giving 8 passages with hybrid search instead of 5 helped clearly (`rag8` better than
  `rag5` on 10 questions, worse on 1). The agent was better than `rag8` on the questions that involve several
  companies (83.3% against 66.7% on 30 questions) at about 2.5 times the time, but with so few questions the
  difference is not statistically clear.
* **What did not help.** Writing tables value by value (row and column) did not improve retrieval or answers over
  plain text (section 7).
* **Main limits.** One grader, small question sets (wide intervals), one language model, one embedding model, one chunk
  size, a 38-filing corpus, and test questions that were read during development (section 9).

## 2. Data

**Source.** SEC EDGAR, the public archive of company filings. The client follows the SEC rules: a User-Agent with a
name and email address (read from an ignored `.env` file, never committed), at most 5 requests per second, and retries
with a wait on errors 429/5xx.

**Companies.** 19 companies, two fiscal years each, 38 filings:

* banks and financials: JPM, BAC, GS, COF, SCHW, TFC
* insurers: MET, AIG, PRU, TRV
* technology: MSFT, GOOGL, AMZN, NVDA, META, AAPL
* others: PFE, JNJ, WMT

Financial companies make up about half of the list; their filings are number-heavy and table-heavy.

**Sections.** Only Item 1 (business), Item 1A (risk factors) and Item 7 (management's discussion and analysis, MD&A)
are kept. Sections are found from the "Item X" headings. Some filers put only a cross-reference under Item 7 and place
the real MD&A in a separate financial section; a fallback looks for the title "Management's Discussion and Analysis"
and stops at the first end marker. The extraction returns warnings instead of failing silently, and repeated page
footers (company name, year, page number) are removed.

**Chunks.** Maximum 1,600 characters (about 400 tokens), 200 characters of overlap, paragraphs are not split, and a
final piece shorter than 300 characters is merged into the previous chunk. Result: 9,592 chunks (business 1,955, risk
factors 3,021, MD&A 4,616). Each chunk has a clean `text` (shown and cited) and an `embed_text` (with company, period
and section in a header, used only to compute the embedding). *The chunk size was not compared with other sizes.*

## 3. Retrieval

* **Embeddings:** `BAAI/bge-m3` (dense, 1,024 dimensions), fp16, maximum length 640 tokens. The longest chunk measured
  633 tokens, so nothing is cut. Vectors are normalised and stored as float16.
* **Index:** exact FAISS `IndexFlatIP`. With about 10,000 vectors an exact search takes milliseconds; an approximate
  index would only add parameters. *Reasoned, not benchmarked.*
* **Scope detection** (`src/retrieval/scope.py`): rules, not a model, find which company and year a question is
  about. If the question does not name a company ("the maker of the iPhone"), no filter is applied: no filter is
  better than a wrong one.
* **Keyword search:** a BM25 written for this project (k1 = 1.5, b = 0.75, with tests). It is used in the hybrid
  setting.
* **Reranker:** cross-encoder `BAAI/bge-reranker-v2-m3` (fp16, length 640). It reads the question and a passage
  together, which is slower but more precise, so it is used only on a candidate pool.

**Ablation on the development questions (71 questions with an answer, no language model).** This analysis was run
during development with the same chunks and is not re-run in this repository; the numbers are kept in the first
version's output.

| setting | hit@5 | all@5 | all@10 | MRR |
|---|---|---|---|---|
| dense (baseline) | 0.732 | 0.549 | 0.676 | 0.533 |
| dense + company/year filter | 0.761 | 0.592 | 0.704 | 0.586 |
| dense + reranker | 0.831 | 0.634 | 0.789 | 0.696 |
| filter + reranker | 0.887 | 0.718 | 0.831 | 0.757 |

`hit@k`: at least one evidence chunk among the top k. `all@k`: all evidence chunks among the top k. The reranker and
the filter each help, and together most. `all@5` stays well below `hit@5`, which is why `rag8` and the agent give
8 passages to the model instead of 5: the evidence of a question often needs several chunks and the fifth place is
too tight. Whether 8 is better is tested on hand-graded answers (section 6), not on this table.

## 4. Generation and the agent

**Language model.** `Qwen/Qwen3.5-9B`, 4-bit NF4 with fp16 compute (the free Colab T4 has no bf16), greedy decoding,
"thinking" off. It uses about 9.6 GB and fits the free T4. A smoke test on 7 questions (not a benchmark) gave 100%
right behaviour for Qwen3.5-9B and 86% for Qwen3-8B at similar speed. *No stronger model was compared.*

**Answer prompt.** Use only the given passages, cite the passage ids in square brackets after each claim, and write
exactly `NOT_FOUND` when the information is not there. Refusals are detected by code, not by reading.

**Agent (`agent_v3`)**: a LangGraph graph with four steps, *plan -> search -> check -> answer*.

1. **Plan.** The model returns a JSON list of searches, one for every company and year that is needed, each with a
   descriptive query. If a question describes a company without naming it, the model picks the ticker from the list.
   If the JSON is invalid, the agent falls back to a plain search with the rule-based scope.
2. **Search.** Each search is restricted to one company and year and takes 4 candidates (6 when there is a single
   search).
3. **Check.** For every company and year needed, the model writes the id of the passage that contains the number, or
   `MISSING`, and then a JSON line saying whether the evidence is enough and, if not, which searches to add (at most
   3). At most 2 rounds of search.
4. **Answer.** The context has 8 passages: the 2 best of every planned search, then filled by ranking the rest with
   the original question.

**How the agent got its design.** An earlier version already planned, searched and checked, but did not improve the
questions with several companies. Reading its traces showed that the plan was right (one search per company and
year) while the passages with the numbers did not reach the model, because everything was ranked with the original
question and a generic passage could beat the one with the number. The fix was to select passages per search and to
add the checklist. The questions of the agentic set guided this diagnosis (see the limitations).

A faster variant (`agent_v3_fast`) skips the check when the plan has a single search. It applies to 89 of the 175
questions and cut the mean time on them from 32.7 to 20.0 seconds. It is not the main configuration.

## 5. Question set and evaluation protocol

**Questions.** 175 questions (`questions/questions.jsonl`), each with a hand-written reference answer and the evidence
chunks. 155 have an answer in the filings and 20 are *unanswerable* (the correct behaviour is to refuse; the absence
was checked with keyword search).

| set | ids | n | fact | numeric | comparison | multi_hop | false_premise | unanswerable |
|---|---|---|---|---|---|---|---|---|
| dev | Q001-Q081 | 81 | 25 | 18 | 13 | 10 | 5 | 10 |
| test | Q082-Q145 | 64 | 10 | 14 | 16 | 8 | 10 | 6 |
| agentic | Q146-Q175 | 30 | 4 | 4 | 12 | 4 | 2 | 4 |

* `fact`: one piece of information. `numeric`: one number. `comparison`: companies or years against each other.
  `multi_hop`: needs several passages. `false_premise`: the question starts from something wrong and the system must
  correct it. `unanswerable`: not in the corpus, the system must refuse.
* The `agentic` set was written for questions about several companies or without a named company (multi-company,
  descriptions, computed values, dependent questions, reverse lookup).
* A validator (`python -m src.eval.golden`) checks the format, unique ids, that every evidence chunk exists and
  belongs to the indicated filing, and that dates and tickers are consistent.

**Automatic checks** (all 175 questions): right behaviour (answers when it should, refuses when it should), valid
citations, whether a citation points to an evidence chunk, retrieval metrics, and whether the key numbers of the
reference answer appear in the answer (an indicator only). There is no language-model judge.

**Hand grading** (94 questions: test and agentic). Each answer is graded `correct`, `partial` or `wrong` against the
reference answer, on sheets where rows are shuffled and the grader does not know which system or chunking produced the
answer. Identical answers share a row and so share the grade. The development set (81 questions) was *not* graded by
hand. There is one grader.

**Statistics.** Wilson 95% intervals for proportions. For comparisons, the paired difference on the same question
(correct 1, partial 0.5, wrong 0) and an exact two-sided sign test. No correction for multiple comparisons.

## 6. Results

All numbers below come from `python -m src.eval.grades_report`. "Strict" counts only `correct`; "lenient" also counts
`partial`.

**All 94 graded questions (text chunks)**

| system | correct | partial | wrong | strict (95% CI) | lenient |
|---|---|---|---|---|---|
| `rag_base` | 67 | 5 | 22 | 71.3% (61-79) | 76.6% |
| `rag5` | 70 | 9 | 15 | 74.5% (65-82) | 84.0% |
| `rag8` | 76 | 8 | 10 | 80.9% (72-88) | 89.4% |
| `agent_v3` | 78 | 7 | 9 | 83.0% (74-89) | 90.4% |

**Test set (64 questions)**

| system | strict (95% CI) | lenient |
|---|---|---|
| `rag_base` | 78.1% (67-86) | 85.9% |
| `rag5` | 79.7% (68-88) | 92.2% |
| `rag8` | 87.5% (77-94) | 95.3% |
| `agent_v3` | 82.8% (72-90) | 93.8% |

**Agentic set (30 questions)**

| system | strict (95% CI) | lenient |
|---|---|---|
| `rag_base` | 56.7% (39-73) | 56.7% |
| `rag5` | 63.3% (46-78) | 66.7% |
| `rag8` | 66.7% (49-81) | 76.7% |
| `agent_v3` | 83.3% (66-93) | 83.3% |

**Fully correct answers by question type (94 questions)**

| type | n | `rag_base` | `rag5` | `rag8` | `agent_v3` |
|---|---|---|---|---|---|
| fact | 14 | 13 | 12 | 13 | 12 |
| numeric | 18 | 17 | 16 | 17 | 17 |
| comparison | 28 | 12 | 13 | 17 | 19 |
| multi_hop | 12 | 6 | 8 | 8 | 8 |
| false_premise | 12 | 9 | 11 | 11 | 12 |
| unanswerable | 10 | 10 | 10 | 10 | 10 |

**Paired comparisons (94 questions; mean score difference, questions better/worse, sign test p)**

| comparison | difference | better / worse | p |
|---|---|---|---|
| `rag5` - `rag_base` | +0.053 | 12 / 7 | 0.36 |
| `rag8` - `rag5` | +0.059 | 10 / 1 | 0.012 |
| `rag8` - `rag_base` | +0.112 | 15 / 3 | 0.008 |
| `agent_v3` - `rag8` | +0.016 | 7 / 4 | 0.55 |
| `agent_v3` - `rag5` | +0.074 | 13 / 4 | 0.049 |
| `agent_v3` - `rag_base` | +0.128 | 17 / 4 | 0.007 |
| agentic set only: `agent_v3` - `rag8` | +0.117 | 6 / 1 | 0.125 |
| agentic set only: `agent_v3` - `rag_base` | +0.267 | 9 / 1 | 0.021 |

**Reading the results**

1. **More passages and hybrid search help.** `rag8` is clearly better than `rag5` and `rag_base`. The filter and
   reranker alone (`rag5` against `rag_base`) are not clearly better on these questions (p = 0.36), even though they
   are much better on the retrieval metrics of section 3. A better ranking does not always change the answer.
2. **The agent is not clearly better than `rag8` overall** (+0.016, p = 0.55). Its advantage is concentrated in
   comparison questions (19 correct against 17) and in the agentic set (83.3% against 66.7%). With 30 questions that
   difference is not statistically clear (p = 0.125 and the intervals overlap).
3. **Time.** Mean seconds per question on a T4, over the 175 questions: `rag_base` 12.6, `rag5` 14.4, `rag8` 15.9,
   `agent_v3` 39.0 (about 2.5 times `rag8`).
4. **Refusals and false premises** are mostly handled by every system: all four refuse correctly on the 10 graded
   unanswerable questions.
5. **Automatic checks (175 questions)**, text chunks:

| system | right behaviour | citation points to evidence | key numbers in answer |
|---|---|---|---|
| `rag_base` | 85.1% | 60.0% | 40.0% |
| `rag5` | 92.0% | 73.5% | 45.8% |
| `rag8` | 92.6% | 77.4% | 47.7% |
| `agent_v3` | 94.3% | 78.1% | 50.3% |

These are indicators, not correctness: the last column in particular counts only the numbers written in the answer.

**Reproducibility.** Running the text variant again in this repository gave exactly the same answers as the first
run for every question and system (greedy decoding, same model and chunks).

## 7. Table-aware chunking: a negative result

SEC filings are full of tables, and flattening them loses the meaning of the columns ("Revenue 391 383"). The
alternative tested here writes every value together with its row and column and makes each table its own chunk,
for example `iPhone: 2026 = $254,098; 2025 = $209,586`. The parser also takes the table title from the table itself,
gives a table split in two the header of the first part, matches rows with fewer values to the year columns, removes
money units from percentage tables, and keeps a short table in the same chunk as the paragraph before it
(`src/ingest/tables.py`).

**Retrieval** (155 answerable questions, no language model). The evidence of the table version is translated from the
text chunks by matching content, so `hit@5` and `recall@5` are approximate. The last column does not depend on that
translation: the share of questions where every number of the reference answer (3 or more digits, not years) appears in
the 5 best passages.

| setting | chunks | hit@5 | recall@5 | numbers in top 5 |
|---|---|---|---|---|
| dense only | text | 0.716 | 0.614 | 0.521 |
| dense only | tables | 0.742 | 0.526 | 0.427 |
| filter + reranker | text | 0.852 | 0.773 | 0.594 |
| filter + reranker | tables | 0.890 | 0.686 | 0.562 |
| filter + hybrid + reranker | text | 0.884 | 0.788 | 0.594 |
| filter + hybrid + reranker | tables | 0.884 | 0.681 | 0.562 |

`hit@5` is equal or slightly higher with tables, but `recall@5` and the number-based measure are lower. In the paired
view the differences on `hit@5` are not significant in any setting.

**Answers** (94 hand-graded questions, text against table-aware chunks):

| system | strict, text | strict, tables | tables minus text (score) | better / worse | p |
|---|---|---|---|---|---|
| `rag_base` | 71.3% | 59.6% | -0.096 | 5 / 15 | 0.041 |
| `rag5` | 74.5% | 73.4% | -0.016 | 7 / 9 | 0.80 |
| `rag8` | 80.9% | 81.9% | -0.005 | 5 / 3 | 0.73 |
| `agent_v3` | 83.0% | 85.1% | +0.021 | 6 / 2 | 0.29 |

For the three stronger systems the difference is within the noise; only the simplest system is clearly worse (and
that is one test among four, uncorrected). By question type, with the four systems pooled, mean scores moved from 0.938
to 0.882 on numeric questions and from 0.771 to 0.698 on multi-hop questions, and were nearly equal on the others.
An earlier version of the table parser (git tag `eval-tables-v1`) was graded as well and gave a similar picture (strict
accuracy 59.6%, 70.2%, 80.9%, 81.9% for the four systems). The final pipeline therefore uses plain-text chunks, and
the table parser is kept as an option (`02b_chunk_and_embed_tables.ipynb`).

The tables did improve a few individual answers (mostly comparison questions). In several of them the model received no
table chunk, so these changes look like effects of a different retrieval rather than of the table format; no cause was
established.

## 8. Where the errors are

An early error analysis (an earlier version of the systems, test questions, 13 of 64 answers not fully correct)
found that most failures were retrieval misses, with the evidence chunk not in the top 5, and that the missed chunks
were mostly numeric tables with little prose around them. Arithmetic was the cause in about one case, so a calculator
tool was judged to be of little value. In the final evaluation the weak points are the same two groups:
comparison questions (12 to 19 correct out of 28) and multi-hop questions (6 to 8 out of 12), which need evidence from
several places. That analysis was not repeated on the final systems.

The agent's check asked for a second round of search in 121 of the 175 questions; it is *not verified* whether those
extra searches were needed.

## 9. Limitations

* **Test questions were read during development.** The test questions were inspected in the error analysis, and the
  agentic set guided the design of the agent. Their results are therefore not a fully blind test.
* **The development set (81 questions) has no hand grades.** Its results are available only from the automatic
  checks.
* **One grader.** Grades were given by one person, blind to the system but not independent. There is no estimate of
  grader agreement.
* **Small sets, wide intervals.** The agentic set has 30 questions; many differences between neighbouring systems are
  within the noise. Several comparisons are reported without correction for multiple tests.
* **One of everything.** One language model (Qwen3.5-9B, 4-bit), one embedding model, one reranker, one chunk size
  (1,600 characters). Other choices could change the ranking of the systems. No stronger or hosted model was tried.
* **Small corpus.** 38 filings from 19 companies, three sections each. The behaviour with a much larger corpus
  (more distractors) is not measured.
* **Approximate evidence translation for tables.** Retrieval metrics for the table-aware chunks rely on translating the
  evidence chunks by text matching; the number-based measure was added because of this.
* **Tables.** The default pipeline reads tables as flattened text. The structured version was not better in these
  tests; improvements to table handling (for example XBRL data or layout-aware parsers) were not tried.
* **Retrieval ablation not re-run.** The table in section 3 comes from the first version of the project, with the same
  text chunks.
* **Automatic indicators.** `numbers_match` and citation checks are indicators of surface properties, not of
  correctness.

## 10. Reproducing the work

The notebooks run in order on the free Colab T4 (secret `HF_TOKEN` is needed; data, embeddings and raw answers are
stored in a private Hugging Face dataset and are not part of this repository):

1. `01_download_data.ipynb`: download the filings from EDGAR (needs the SEC User-Agent in a Colab secret or `.env`).
2. `02a_chunk_and_embed_text.ipynb`: text chunks and embeddings (the pipeline used for all main results).
3. `02b_chunk_and_embed_tables.ipynb`: table-aware chunks and embeddings (the comparison of section 7).
4. `03_compare_text_vs_tables.ipynb`: retrieval comparison between text and table-aware chunks.
5. `04_run_all_systems.ipynb`: the four systems on all 175 questions, once per variant (`VARIANT = "text"` or
   `"tables"`); answers are saved to the dataset every 20 questions, so a stopped run can resume from another
   account.
6. Hand grading on the shuffled sheets, then `python -m src.eval.grades_report` for the numbers in this report.

Tests and style checks (`pytest`, `ruff`) run in the CI. The grading sheets, their keys and the retrieval comparison
are in `results/`.

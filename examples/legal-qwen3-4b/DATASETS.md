All 48 datasets are accounted for below. No new custom adapters are required for the 13 selected sources.

| Dataset | Decision | Reason |
|---|---|---|
| `FiscalNote/billsum` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `TheTokenFactory/sec-contracts-financial-extraction-instructions` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `ymoslem/Law-StackExchange` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `AdaptLLM/law-tasks` | evaluation_only | Reserved for evaluation; never training. |
| `Alignment-Lab-AI/Lawyer-Instruct` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `ChicagoHAI/CaseSumm` | needs_wrapper_or_policy | Conflicting license statements; resolve before mixture admission. |
| `FredrikBL/Legal-Snigel-DPO` | needs_wrapper_or_policy | Preference objective and missing license; excluded from SFT. |
| `HFforLegal/case-law` | cpt_candidate | Raw text is a separate continued-pretraining objective, outside default SFT. |
| `Prarabdha/indian-legal-supervised-fine-tuning-data` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `ShengbinYue/DISC-Law-SFT` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `a2aj/canadian-case-law` | needs_wrapper_or_policy | Per-row non-commercial source terms; requires explicit source policy. |
| `casehold/casehold` | needs_source_terms | Source terms absent from card; withheld from default mixture. |
| `chenghao/sec-material-contracts` | cpt_candidate | Raw text is a separate continued-pretraining objective, outside default SFT. |
| `china-ai-law-challenge/cail2018` | needs_wrapper_or_policy | Needs split-family selection and fact-level decontamination. |
| `coastalcph/lex_glue` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `coastalcph/multi_eurlex` | needs_wrapper_or_policy | Needs a pinned direct archive loader and resolved label hierarchy. |
| `common-pile/caselaw_access_project` | cpt_candidate | Raw text is a separate continued-pretraining objective, outside default SFT. |
| `crosbylegal/RedlineBench` | evaluation_only | Reserved for evaluation; never training. |
| `dennlinger/eur-lex-sum` | needs_wrapper_or_policy | Needs direct language-file loader; conflicting license statements. |
| `doolayer/LawBench` | evaluation_only | Reserved for evaluation; never training. |
| `dzunggg/legal-qa-v1` | needs_source_terms | Source terms absent from card; withheld from default mixture. |
| `hails/agieval-jec-qa-kd` | evaluation_only | Reserved for evaluation; never training. |
| `harveyai/harvey-labs` | evaluation_only | Reserved for evaluation; never training. |
| `isaacus/legal-rag-bench` | evaluation_only | Reserved for evaluation; never training. |
| `isaacus/open-australian-legal-corpus` | cpt_candidate | Raw text is a separate continued-pretraining objective, outside default SFT. |
| `isaacus/open-australian-legal-qa` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `jhu-clsp/CLERC` | needs_wrapper_or_policy | Requires generation/train.jsonl mapping and source terms; not retrieval rows. |
| `joelniklaus/Multi_Legal_Pile` | needs_wrapper_or_policy | Unpinned upstream downloads and merged Pile-of-Law validation; needs direct-file loader. |
| `kiddothe2b/contract-nli` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `lawinstruct/lawinstruct` | needs_wrapper_or_policy | Needs explicit file allowlist; several files copy benchmark families. |
| `lighteval/legal_summarization` | evaluation_only | Reserved for evaluation; never training. |
| `louisbrulenaudet/legalkit` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `mb7419/legal-advice-reddit_preference` | needs_wrapper_or_policy | Preference objective and missing license; excluded from SFT. |
| `nguha/legalbench` | evaluation_only | Reserved for evaluation; never training. |
| `nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `nvidia/Nemotron-Pretraining-Legal-v1` | cpt_candidate | Raw text is a separate continued-pretraining objective, outside default SFT. |
| `open-agreements/legal-practice-library` | cpt_candidate | Raw text is a separate continued-pretraining objective, outside default SFT. |
| `pile-of-law/pile-of-law` | needs_wrapper_or_policy | Needs direct-file loader and benchmark decontamination before CPT. |
| `reglab/barexam_qa` | needs_wrapper_or_policy | Needs direct CSV mapping preserving prompt_id shared contexts. |
| `reglab/housing_qa` | evaluation_only | Reserved for evaluation; never training. |
| `reglab/legal_rag_hallucinations` | evaluation_only | Reserved for evaluation; never training. |
| `ricdomolm/caselawqa-8k` | evaluation_only | Reserved for evaluation; never training. |
| `ricdomolm/lawma-instructions` | needs_source_terms | Source terms absent from card; withheld from default mixture. |
| `ricdomolm/lawma-tasks` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `stindardlogic/legal-reasoning-dpo-100k` | needs_wrapper_or_policy | Preference objective, only 16 unique pairs; excluded from SFT. |
| `theatticusproject/cuad` | needs_wrapper_or_policy | Raw SQuAD release; use cuad-qa rather than auto-detected text lines. |
| `theatticusproject/cuad-qa` | ready: cached rows | Pinned loader and explicit adapter; must pass real-row preparation. |
| `theatticusproject/maud` | needs_wrapper_or_policy | Needs contract-level resplit; published test contracts all occur in train. |

"""Auditable recipes. No dataset repository's Python code is executed."""

from typing import Any

PILOT = [
    "FiscalNote/billsum",
    "TheTokenFactory/sec-contracts-financial-extraction-instructions",
    "ymoslem/Law-StackExchange",
]
SFT = {
    "Alignment-Lab-AI/Lawyer-Instruct": (["instruction", "input"], "output"),
    "dzunggg/legal-qa-v1": (["question"], "answer"),
    "ricdomolm/lawma-instructions": (["instruction"], "output"),
    "isaacus/open-australian-legal-qa": (["question"], "answer"),
    "nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions": (
        ["instruction", "input"],
        "output",
    ),
    "louisbrulenaudet/legalkit": (["query"], "output"),
    "Prarabdha/indian-legal-supervised-fine-tuning-data": (
        ["context", "question"],
        "response",
    ),
    "lawinstruct/lawinstruct": (["instruction", "prompt"], "answer"),
    "ShengbinYue/DISC-Law-SFT": (["input"], "output"),
    "theatticusproject/maud": (["text", "question", "subquestion"], "answer"),
}
TEXT = {
    "pile-of-law/pile-of-law": "text",
    "joelniklaus/Multi_Legal_Pile": "text",
    "HFforLegal/case-law": "document",
    "a2aj/canadian-case-law": "unofficial_text_en",
    "nvidia/Nemotron-Pretraining-Legal-v1": "text",
    "common-pile/caselaw_access_project": "text",
    "isaacus/open-australian-legal-corpus": "text",
    "chenghao/sec-material-contracts": "file_content",
    "open-agreements/legal-practice-library": "markdown",
}
# These require an explicit source policy or specialized objective before admission.
BLOCKED = {
    "joelniklaus/Multi_Legal_Pile": "Unpinned upstream downloads and merged Pile-of-Law validation; needs direct-file loader.",
    "pile-of-law/pile-of-law": "Needs direct-file loader and benchmark decontamination before CPT.",
    "lawinstruct/lawinstruct": "Needs explicit file allowlist; several files copy benchmark families.",
    "theatticusproject/cuad": "Raw SQuAD release; use cuad-qa rather than auto-detected text lines.",
    "theatticusproject/maud": "Needs contract-level resplit; published test contracts all occur in train.",
    "china-ai-law-challenge/cail2018": "Needs split-family selection and fact-level decontamination.",
    "coastalcph/multi_eurlex": "Needs a pinned direct archive loader and resolved label hierarchy.",
    "reglab/barexam_qa": "Needs direct CSV mapping preserving prompt_id shared contexts.",
    "dennlinger/eur-lex-sum": "Needs direct language-file loader; conflicting license statements.",
    "ChicagoHAI/CaseSumm": "Conflicting license statements; resolve before mixture admission.",
    "jhu-clsp/CLERC": "Requires generation/train.jsonl mapping and source terms; not retrieval rows.",
    "stindardlogic/legal-reasoning-dpo-100k": "Preference objective, only 16 unique pairs; excluded from SFT.",
    "mb7419/legal-advice-reddit_preference": "Preference objective and missing license; excluded from SFT.",
    "FredrikBL/Legal-Snigel-DPO": "Preference objective and missing license; excluded from SFT.",
    "a2aj/canadian-case-law": "Per-row non-commercial source terms; requires explicit source policy.",
}


def recipes(card: Any, all_configs: Any = False):
    rid = card["id"]
    base = {"id": rid, "revision": card["commit"], "config": None, "split": "train"}
    if rid in SFT:
        inputs, target = SFT[rid]
        base.update(adapter="sft", inputs=inputs, target=target)
    elif rid in TEXT:
        base.update(
            adapter="text",
            field=TEXT[rid],
            markup=rid == "chenghao/sec-material-contracts",
        )
    elif rid == "FiscalNote/billsum":
        base.update(adapter="summarization", field="text", target="summary")
    elif rid == "ChicagoHAI/CaseSumm":
        base.update(adapter="summarization", field="opinion", target="syllabus")
    elif rid in {"ricdomolm/lawma-tasks", "coastalcph/lex_glue", "casehold/casehold"}:
        base["adapter"] = "mcq"
    elif rid == "TheTokenFactory/sec-contracts-financial-extraction-instructions":
        base.update(adapter="chat", config="sharegpt")
    elif rid == "ymoslem/Law-StackExchange":
        base["adapter"] = "stackexchange"
    elif rid == "theatticusproject/cuad-qa":
        base.update(
            adapter="cuad",
            upstream_zip={
                "url": "https://raw.githubusercontent.com/TheAtticusProject/cuad/7aad649183846688381a147eeb3601aff590fe4a/data.zip",
                "commit": "7aad649183846688381a147eeb3601aff590fe4a",
                "sha256": "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a",
            },
        )
    elif rid == "kiddothe2b/contract-nli":
        base.update(
            adapter="nli",
            config="contractnli_a",
            zip_file="contract_nli.zip",
            zip_member="contract_nli_v1.jsonl",
            validation_split="validation",
        )
    else:
        return []
    if rid == "isaacus/open-australian-legal-qa":
        base.update(
            adapter="grounded_qa",
            direct={"builder": "json", "files": {"train": "qa.jsonl"}},
        )
    if rid == "HFforLegal/case-law":
        base["split"] = "us"
    if rid == "isaacus/open-australian-legal-corpus":
        base.update(config="corpus", split="corpus")
    if rid == "nvidia/Nemotron-Pretraining-Legal-v1":
        base["config"] = "Nemotron-Pretraining-Legal-eCFR"
    if rid == "ShengbinYue/DISC-Law-SFT":
        base["data_files"] = "DISC-Law-SFT-Pair.jsonl"
    if rid == "casehold/casehold":
        base.update(
            config="all",
            direct={
                "builder": "csv",
                "files": {
                    "train": "data/all/train.csv",
                    "validation": "data/all/val.csv",
                },
            },
            validation_split="validation",
        )
    if rid == "ricdomolm/lawma-tasks":
        base.update(config="sc_adminaction", validation_split="val")
        if all_configs:
            from datasets import get_dataset_config_names

            configs = get_dataset_config_names(
                rid, revision=base["revision"], trust_remote_code=False
            )
            return [{**base, "config": c} for c in configs]
    if rid == "coastalcph/lex_glue":
        configs = (
            [
                "case_hold",
                "ecthr_a",
                "ecthr_b",
                "eurlex",
                "ledgar",
                "scotus",
                "unfair_tos",
            ]
            if all_configs
            else ["case_hold", "unfair_tos"]
        )
        return [
            {
                **base,
                "config": c,
                "adapter": "mcq" if c == "case_hold" else "classification",
                "validation_split": "validation",
            }
            for c in configs
        ]
    if rid == "ricdomolm/lawma-instructions":
        base["validation_split"] = "validation"
    return [base]


def decision(card: Any, include_cpt: Any = False):
    if card["use"] == "eval":
        return "evaluation_only", "Reserved for evaluation; never training."
    if card["id"] in BLOCKED:
        return "needs_wrapper_or_policy", BLOCKED[card["id"]]
    if card["flag"] == "no-licence":
        return (
            "needs_source_terms",
            "Source terms absent from card; withheld from default mixture.",
        )
    if card["kind"] == "text" and not include_cpt:
        return (
            "cpt_candidate",
            "Raw text is a separate continued-pretraining objective, outside default SFT.",
        )
    if not recipes(card):
        return "needs_wrapper", "No verified conversion recipe yet."
    return (
        "candidate",
        "Pinned loader and explicit adapter; must pass real-row preparation.",
    )

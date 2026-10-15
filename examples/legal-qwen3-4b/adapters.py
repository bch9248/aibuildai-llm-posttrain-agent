"""Strict legal-row conversion. Never infer a task by dumping the entire row."""

from typing import Any
import hashlib
import html
import json
import re
from html.parser import HTMLParser


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag: Any, attrs: Any):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "br", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: Any):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: Any):
        if not self.hidden:
            self.parts.append(data)


def clean(value: Any, markup: Any = False):
    if isinstance(value, list):
        value = "\n".join(str(x) for x in value)
    if not isinstance(value, str):
        raise ValueError(f"Expected text, got {type(value).__name__}")
    if markup:
        parser = TextOnly()
        parser.feed(value)
        value = "".join(parser.parts)
    return (
        re.sub(r"[ \t]+", " ", html.unescape(value))
        .replace("``", '"')
        .replace("''", '"')
        .strip()
    )


def digest(value: Any):
    return hashlib.sha256(re.sub(r"\s+", " ", value).strip().encode()).hexdigest()


def convert(row: Any, spec: Any):
    """Return canonical messages/text with an input-only document group for splitting."""
    kind = spec["adapter"]
    group = ""
    if kind == "text":
        if row.get("extension") == "pdf":
            return None
        if (
            spec["id"] == "isaacus/open-australian-legal-corpus"
            and row.get("type") == "decision"
        ):
            return None
        text = clean(row[spec["field"]], spec.get("markup", False))
        if not text:
            return None
        return {"text": text, "group": digest(text), "objective": "cpt"}
    if kind == "chat":
        conv = row.get("conversations", row.get("messages"))
        if not isinstance(conv, list):
            raise ValueError("Missing conversation list")
        roles = {
            "human": "user",
            "gpt": "assistant",
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        }
        messages = []
        for m in conv:
            role = roles.get(m.get("role", m.get("from")))
            content = m.get("content", m.get("value"))
            if role is None:
                raise ValueError("Unsupported conversation role")
            messages.append({"role": role, "content": clean(content)})
        group = (
            row.get("metadata", {}).get("source_file", "")
            if isinstance(row.get("metadata"), dict)
            else ""
        )
    elif kind == "stackexchange":
        answers = [a for a in row["answers"] if a.get("body") and a.get("score", 0) > 0]
        if not answers:
            return None
        prompt = (
            clean(row["question_title"]) + "\n\n" + clean(row["question_body"], True)
        )
        target = clean(max(answers, key=lambda a: a["score"])["body"], True)
        group = str(row["question_id"])
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    elif kind == "mcq":
        if spec["id"] == "casehold/casehold":
            context, choices, answer = (
                row["citing_prompt"],
                [row[f"holding_{i}"] for i in range(5)],
                int(row["label"]),
            )
        elif "endings" in row:
            context, choices, answer = row["context"], row["endings"], row["label"]
        else:
            context, choices, answer = row["opinion"], row["choices"], row["answer"]
        answer = answer if isinstance(answer, list) else [answer]
        if (
            not choices
            or not answer
            or any(type(i) is not int or not 0 <= i < len(choices) for i in answer)
        ):
            raise ValueError("Invalid choice indices")
        # Decimal indices work for Lawma's >26 options and multiple correct answers.
        prompt = "\n\n".join(
            x
            for x in [
                clean(context),
                clean(row.get("instruction", "Select the correct holding.")),
                clean(row.get("question", "")),
                "\n".join(f"{i}: {clean(c)}" for i, c in enumerate(choices)),
                "Return the correct zero-based choice indices as a JSON list.",
            ]
            if x
        )
        target = json.dumps(sorted(set(answer)))
        group = clean(context)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    elif kind == "classification":
        text = clean(row["text"])
        labels = row.get("labels", row.get("label"))
        names = spec.get("label_names")
        if labels is None:
            raise ValueError("Missing labels")
        if isinstance(labels, list):
            target = (
                json.dumps([names[x] if names else x for x in labels])
                if labels
                else "none"
            )
        else:
            target = str(names[labels] if names else labels)
        prompt = f"Classify this legal text for the {spec.get('config', '')} task. Labels: {json.dumps(names) if names else 'dataset label IDs'}.\n\n{text}"
        group = text
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    elif kind == "grounded_qa":
        group = clean(row["source"]["text"])
        prompt = (
            "Answer from the supplied Australian legal source.\n\n"
            + group
            + "\n\n"
            + clean(row["question"])
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": clean(row["answer"])},
        ]
    elif kind == "nli":
        group = clean(row["premise"])
        prompt = f"Classify the hypothesis as contradiction, entailment, or neutral.\n\nContract: {group}\n\nHypothesis: {clean(row['hypothesis'])}"
        target = ["contradiction", "entailment", "neutral"][row["label"]]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    elif kind == "cuad":
        group = clean(row["context"])
        prompt = (
            'Extract the requested clause. If absent, answer "no such clause".\n\n'
            + group
            + "\n\n"
            + clean(row["question"])
        )
        answers = row["answers"]["text"]
        target = "\n".join(answers) if answers else "no such clause"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    elif kind == "summarization":
        group = clean(row[spec["field"]])
        messages = [
            {
                "role": "user",
                "content": "Summarize the following legal document.\n\n" + group,
            },
            {"role": "assistant", "content": clean(row[spec["target"]])},
        ]
    elif kind == "sft":
        if spec["id"] == "ShengbinYue/DISC-Law-SFT" and str(
            row.get("id", "")
        ).startswith("exam-"):
            return None
        prompt = "\n\n".join(
            clean(row[x], spec.get("markup", False))
            for x in spec["inputs"]
            if row.get(x)
        )
        target = clean(row[spec["target"]])
        # Group grounded examples by their supplied context; preserve document boundaries.
        group = next(
            (
                clean(row[x])
                for x in ("context", "text", "prompt")
                if row.get(x) and x in spec["inputs"]
            ),
            "",
        )
        if spec["id"] == "theatticusproject/maud":
            group = row["contract_name"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    else:
        raise ValueError(f"Unsupported adapter: {kind}")
    if (
        len(messages) < 2
        or messages[-1]["role"] != "assistant"
        or not any(m["role"] == "user" for m in messages)
    ):
        raise ValueError("Conversation needs a user and a final assistant target")
    if any(not m["content"] for m in messages):
        raise ValueError("Empty message")
    if not group:
        group = "\n".join(m["content"] for m in messages if m["role"] != "assistant")
    return {"messages": messages, "group": digest(str(group)), "objective": "sft"}

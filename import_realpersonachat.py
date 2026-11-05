"""Import local RealPersonaChat utterance JSON into LLM_TMP dialogue examples.

Source schema: dialogue_id, interlocutors, utterances[{utterance_id,
interlocutor_id,text,timestamp}], evaluations. Speaker IDs are local to
each dialogue; no assumption that AA is 'user' or AB is 'assistant'.

Default: validate/audit only. --write exports train/dev/test JSONL in
a separate local directory. Never edits source files or exports to GitHub.
Data ownership/permission and personal-data review are the operator's
responsibility BEFORE training. No intent labels are fabricated.

Split is deterministic by entire conversation, with exact duplicate
dialogues assigned to the same split. This is corpus-heldout only,
NOT an independent new-source evaluation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def read_dialogue(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def validate_dialogue(doc):
    if not isinstance(doc, dict):
        raise ValueError("NOT_OBJECT")
    dialogue_id = doc.get("dialogue_id")
    if not isinstance(dialogue_id, (str, int)) or isinstance(dialogue_id, bool):
        raise ValueError("INVALID_DIALOGUE_ID")
    people = doc.get("interlocutors")
    if (not isinstance(people, list) or len(people) != 2
            or not all(isinstance(p, str) and p.strip() for p in people)
            or len(set(people)) != 2):
        raise ValueError("INVALID_INTERLOCUTORS")
    turns = doc.get("utterances")
    if not isinstance(turns, list) or len(turns) < 2:
        raise ValueError("INSUFFICIENT_TURNS")
    cleaned, ids = [], set()
    speaker_ids = {key: "speaker_" + str(i) for i, key in enumerate(people)}
    for row in turns:
        if not isinstance(row, dict):
            raise ValueError("INVALID_UTTERANCE")
        uid = row.get("utterance_id")
        if not isinstance(uid, int) or isinstance(uid, bool) or uid in ids:
            raise ValueError("INVALID_UTTERANCE_ID")
        ids.add(uid)
        speaker = row.get("interlocutor_id")
        if speaker not in speaker_ids:
            raise ValueError("UNKNOWN_INTERLOCUTOR")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("EMPTY_UTTERANCE")
        cleaned.append({"id": uid, "speaker": speaker_ids[speaker],
                        "text": text.strip()})
    # Use explicit utterance_id order, not timestamps or file order.
    cleaned.sort(key=lambda row: row["id"])
    # The original participant identifiers are never written to outputs.
    return str(dialogue_id), cleaned


def content_fingerprint(turns):
    canonical = json.dumps(
        [(row["speaker"], row["text"]) for row in turns],
        ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assign_split(fingerprint, seed=42, train_fraction=.90, dev_fraction=.05):
    if train_fraction <= 0 or dev_fraction <= 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("INVALID_SPLIT_FRACTIONS")
    h = hashlib.sha256((str(seed) + ":" + fingerprint).encode("ascii")).digest()
    bucket = int.from_bytes(h[:8], "big") / (1 << 64)
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + dev_fraction:
        return "dev"
    return "test"


def iter_examples(dialogue_id, turns, context_turns=4, fingerprint=None):
    if context_turns < 1:
        raise ValueError("CONTEXT_TURNS_MUST_BE_POSITIVE")
    for i in range(1, len(turns)):
        context = turns[max(0, i - context_turns):i]
        target = turns[i]
        yield {
            "dialogue_id": dialogue_id,
            "conversation_fingerprint": fingerprint,
            "turn_id": target["id"],
            "context": [{"speaker": t["speaker"], "text": t["text"]}
                        for t in context],
            "next_speaker": target["speaker"],
            "response": target["text"],
            "speaker_semantics": "two_interlocutors_no_user_assistant_assumption",
            "intent_label": None,
            "fact_verified": False,
            "source": "real_persona_chat",
        }


def convert_directory(source, output=None, seed=42, limit_files=None,
                      context_turns=4, write=False, strict=False):
    source = Path(source)
    if not source.is_dir():
        raise ValueError("SOURCE_DIRECTORY_MISSING")
    paths = sorted(source.glob("*.json"))
    if limit_files is not None:
        if limit_files < 1:
            raise ValueError("INVALID_LIMIT_FILES")
        paths = paths[:limit_files]
    if not paths:
        raise ValueError("NO_JSON_FILES")
    output_path = Path(output) if output is not None else None
    if write and output_path is None:
        raise ValueError("OUTPUT_REQUIRED_FOR_WRITE")
    if write and (output_path.resolve() == source.resolve()
                  or source.resolve() in output_path.resolve().parents):
        raise ValueError("OUTPUT_MUST_BE_OUTSIDE_SOURCE")
    if write and any(output_path.resolve() in path.resolve().parents
                     for path in paths):
        raise ValueError("OUTPUT_OVERLAPS_SOURCE")
    # Preflight whole set before writing, so malformed JSON doesn't leave
    # partially generated datasets. With 13k files, memory is still modest:
    # store dialogue objects for streamed output pass after preflight.
    accepted = []
    errors = collections.Counter()
    first_error = None
    duplicate_ids = set()
    seen_ids = set()
    unique_fingerprints = set()
    files_by_split = collections.Counter()
    turns_by_split = collections.Counter()
    examples_by_split = collections.Counter()
    duplicate_dialogues = 0
    for path in paths:
        try:
            identifier, turns = validate_dialogue(read_dialogue(path))
            if identifier in seen_ids:
                raise ValueError("DUPLICATE_DIALOGUE_ID")
            seen_ids.add(identifier)
            fingerprint = content_fingerprint(turns)
            if fingerprint in unique_fingerprints:
                duplicate_dialogues += 1
                # Ignore repeat content; all identical content belongs to
                # a single split, and only one copy is emitted.
                continue
            unique_fingerprints.add(fingerprint)
            split = assign_split(fingerprint, seed)
            accepted.append((path, identifier, fingerprint, split))
            files_by_split[split] += 1
            turns_by_split[split] += len(turns)
            examples_by_split[split] += len(turns) - 1
        except (ValueError, json.JSONDecodeError, UnicodeError) as exc:
            key = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            errors[key] += 1
            if first_error is None:
                first_error = {"filename": path.name, "error": key}
            if strict:
                raise ValueError("INVALID_SOURCE: " + path.name + ": " + key) from exc

    result = {
        "status": "AUDIT_ONLY" if not write else "EXPORTED_CORPUS_HELDOUT",
        "source_files_scanned": len(paths),
        "unique_dialogues": len(accepted),
        "duplicate_content_dialogues_skipped": duplicate_dialogues,
        "invalid_files": sum(errors.values()),
        "invalid_reasons": dict(errors),
        "first_error": first_error,
        "source_schema": "real_persona_chat_v1",
        "split_basis": "sha256_full_dialogue_content",
        "split_seed": seed,
        "split_fractions": {"train": .90, "dev": .05, "test": .05},
        "dialogues_by_split": dict(files_by_split),
        "utterances_by_split": dict(turns_by_split),
        "examples_by_split": dict(examples_by_split),
        "speaker_assumption": "NONE",
        "intent_labels": "NOT_PROVIDED",
        "evaluation_limit": "corpus-heldout; NOT independent new-source",
    }
    if write:
        output_path.mkdir(parents=True, exist_ok=True)
        # Only create outputs after all checked. Reject pre-existing output
        # files rather than clobbering an earlier split/run.
        names = [output_path / (split + ".jsonl")
                 for split in ("train", "dev", "test")]
        names.append(output_path / "audit.json")
        if any(path.exists() for path in names):
            raise ValueError("OUTPUT_EXISTS_USE_FRESH_DIRECTORY")
        handles = {split: (output_path / (split + ".jsonl")).open(
            "x", encoding="utf-8") for split in ("train", "dev", "test")}
        try:
            for path, identifier, fingerprint, split in accepted:
                _, turns = validate_dialogue(read_dialogue(path))
                for example in iter_examples(identifier, turns, context_turns,
                                             fingerprint):
                    handles[split].write(
                        json.dumps(example, ensure_ascii=False) + "\n")
        finally:
            for handle in handles.values():
                handle.close()
        (output_path / "audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="data/realpersonachat_import")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-files", type=int)
    p.add_argument("--context-turns", type=int, default=4)
    p.add_argument("--write", action="store_true",
                   help="Explicitly allow local text export (default audit only)")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    report = convert_directory(
        args.source, args.output, seed=args.seed,
        limit_files=args.limit_files, context_turns=args.context_turns,
        write=args.write, strict=args.strict)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

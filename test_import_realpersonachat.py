"""Tests for importing actual RealPersonaChat JSON dialogue structure."""
import json
import tempfile
import unittest
from pathlib import Path

from import_realpersonachat import (
    assign_split, content_fingerprint, convert_directory,
    iter_examples, validate_dialogue,
)


def sample(dialogue_id=1, text="よろしくお願いいたします。"):
    return {
        "dialogue_id": dialogue_id,
        "interlocutors": ["AA", "AB"],
        "utterances": [
            {"utterance_id": 2, "interlocutor_id": "AA",
             "text": "今日は涼しいですね", "timestamp": "2022-08-06T14:51:55"},
            {"utterance_id": 0, "interlocutor_id": "AA",
             "text": text, "timestamp": "2022-08-06T14:51:18"},
            {"utterance_id": 1, "interlocutor_id": "AB",
             "text": "よろしくお願いします！", "timestamp": "2022-08-06T14:51:48"},
            {"utterance_id": 3, "interlocutor_id": "AB",
             "text": "雨が降って涼しいですね。", "timestamp": "2022-08-06T14:52:07"},
        ],
        "evaluations": [{"interlocutor_id": "AA", "satisfaction": 5}],
    }


class ImportTests(unittest.TestCase):
    def test_valid_and_sorted(self):
        identity, rows = validate_dialogue(sample())
        self.assertEqual(identity, "1")
        self.assertEqual([r["id"] for r in rows], [0, 1, 2, 3])

    def test_speaker_normalized_no_bot_assumption(self):
        _, rows = validate_dialogue(sample())
        self.assertEqual([r["speaker"] for r in rows],
                         ["speaker_0", "speaker_1", "speaker_0", "speaker_1"])

    def test_one_target_per_nonfirst_utterance(self):
        _, rows = validate_dialogue(sample())
        examples = list(iter_examples("1", rows, context_turns=2))
        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[-1]["next_speaker"], "speaker_1")
        self.assertEqual(len(examples[-1]["context"]), 2)
        self.assertIsNone(examples[0]["intent_label"])

    def test_reject_unknown_speaker(self):
        d = sample()
        d["utterances"][0]["interlocutor_id"] = "ZZ"
        with self.assertRaisesRegex(ValueError, "UNKNOWN_INTERLOCUTOR"):
            validate_dialogue(d)

    def test_reject_duplicate_turn(self):
        d = sample()
        d["utterances"][1]["utterance_id"] = 2
        with self.assertRaisesRegex(ValueError, "INVALID_UTTERANCE_ID"):
            validate_dialogue(d)

    def test_reject_empty_turn(self):
        d = sample()
        d["utterances"][1]["text"] = ""
        with self.assertRaisesRegex(ValueError, "EMPTY_UTTERANCE"):
            validate_dialogue(d)

    def test_reject_missing_people(self):
        d = sample()
        d["interlocutors"] = ["AA"]
        with self.assertRaisesRegex(ValueError, "INVALID_INTERLOCUTORS"):
            validate_dialogue(d)

    def test_deterministic_split(self):
        _, rows = validate_dialogue(sample())
        h = content_fingerprint(rows)
        self.assertEqual(assign_split(h), assign_split(h))

    def test_identical_dialogue_hash(self):
        _, rows = validate_dialogue(sample())
        d = sample(dialogue_id=44)
        _, rows2 = validate_dialogue(d)
        self.assertEqual(content_fingerprint(rows),
                         content_fingerprint(rows2))

    def test_audit_without_text_export(self):
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "input", Path(td) / "output"
            inp.mkdir()
            (inp / "00001.json").write_text(json.dumps(sample(),
                         ensure_ascii=False), encoding="utf-8")
            report = convert_directory(inp, out)
            self.assertEqual(report["status"], "AUDIT_ONLY")
            self.assertEqual(report["unique_dialogues"], 1)
            self.assertFalse(out.exists())

    def test_explicit_write_and_preserve_content(self):
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "input", Path(td) / "output"
            inp.mkdir()
            (inp / "00001.json").write_text(json.dumps(sample(),
                         ensure_ascii=False), encoding="utf-8")
            report = convert_directory(inp, out, write=True)
            self.assertEqual(report["status"], "EXPORTED_CORPUS_HELDOUT")
            files = [out / (name + ".jsonl") for name in
                     ("train", "dev", "test")]
            lines = [json.loads(line) for file in files for line in
                     file.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), 3)
            self.assertTrue(all(x["intent_label"] is None for x in lines))
            self.assertFalse(any("AA" in json.dumps(x) for x in lines))

    def test_duplicate_content_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td)
            for i in (1, 2):
                (src / ("%05d.json" % i)).write_text(
                    json.dumps(sample(dialogue_id=i), ensure_ascii=False),
                    encoding="utf-8")
            r = convert_directory(src)
            self.assertEqual(r["unique_dialogues"], 1)
            self.assertEqual(r["duplicate_content_dialogues_skipped"], 1)

    def test_bad_json_counted(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td)
            (src / "00001.json").write_text("{bad", encoding="utf-8")
            report = convert_directory(src)
            self.assertEqual(report["invalid_files"], 1)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input"
            out = Path(td) / "out"
            inp.mkdir()
            (inp / "00001.json").write_text(
                json.dumps(sample(), ensure_ascii=False), encoding="utf-8")
            convert_directory(inp, out, write=True)
            with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS"):
                convert_directory(inp, out, write=True)


if __name__ == "__main__":
    unittest.main()

"""Small, fully visible Japanese dialogue-act development fixtures.

Each intent is split by phrase families (not exact duplicated strings).
The split is fixed, reproducible, and used only for development. Because
both vocab and pragmatic categories are hand-designed, these examples
cannot serve as an independent/general-purpose Japanese holdout.
"""
from __future__ import annotations

import random

# Each family contains surface variants of one narrow pragmatic situation.
# Family-level split prevents exact rephrasings of the same example in dev.
FAMILIES = {
    "GREETING": [
        ["こんにちは。", "こんにちは！"],
        ["おはよう。", "おはようございます。"],
        ["こんばんは。", "こんばんは！"],
        ["やあ。", "やあ、どうも。"],
        ["久しぶり。", "お久しぶりです。"],
        ["はじめまして。", "はじめまして！"],
    ],
    "FAREWELL": [
        ["さようなら。", "さようなら！"],
        ["またね。", "じゃあまたね。"],
        ["おやすみ。", "おやすみなさい。"],
        ["また明日。", "ではまた明日。"],
        ["失礼します。", "それでは失礼します。"],
        ["そろそろ行きます。", "もう行きますね。"],
    ],
    "THANKS": [
        ["ありがとう。", "ありがとうございます。"],
        ["助かったよ。", "とても助かりました。"],
        ["感謝します。", "心から感謝します。"],
        ["どうもありがとう。", "本当にありがとう。"],
        ["教えてくれてありがとう。", "説明してくれてありがとう。"],
        ["いつもありがとう。", "今日もありがとう。"],
    ],
    "QUESTION": [
        ["何をしていますか？", "今何をしていますか？"],
        ["これは何ですか？", "それは何ですか？"],
        ["どうしてですか？", "どうしてそうなるの？"],
        ["どこにありますか？", "どこにあるの？"],
        ["今日は何曜日ですか？", "明日は何曜日ですか？"],
        ["この意味は何ですか？", "その意味は何ですか？"],
    ],
    "REQUEST": [
        ["説明してください。", "詳しく説明してください。"],
        ["教えてください。", "もう少し教えてください。"],
        ["手伝ってください。", "少し手伝ってください。"],
        ["まとめてください。", "短くまとめてください。"],
        ["もう一度お願いします。", "もう一度説明をお願いします。"],
        ["確認してください。", "内容を確認してください。"],
    ],
    "SELF_DISCLOSURE": [
        ["今日は仕事でした。", "今日は朝から仕事でした。"],
        ["最近、本を読んでいます。", "最近はよく本を読んでいます。"],
        ["私は東京に住んでいます。", "私は今東京に住んでいます。"],
        ["昨日散歩しました。", "昨日は少し散歩しました。"],
        ["私は猫が好きです。", "私は猫がとても好きです。"],
        ["週末は家にいました。", "週末はずっと家にいました。"],
    ],
    "FEELING": [
        ["嬉しいです。", "とても嬉しいです。"],
        ["悲しいです。", "少し悲しいです。"],
        ["不安です。", "今日は少し不安です。"],
        ["楽しいです。", "本当に楽しいです。"],
        ["眠いです。", "とても眠いです。"],
        ["怖いです。", "少し怖いです。"],
    ],
    "ACKNOWLEDGEMENT": [
        ["はい。", "はい、そうです。"],
        ["わかりました。", "よくわかりました。"],
        ["なるほど。", "なるほど、そうですね。"],
        ["そうなんですね。", "そうだったんですね。"],
        ["了解です。", "了解しました。"],
        ["ええ。", "ええ、そうです。"],
    ],
}


def build_intent_examples(seed=42):
    rng = random.Random(seed)
    train, dev = [], []
    for intent, families in FAMILIES.items():
        order = list(range(len(families)))
        rng.shuffle(order)
        # Four families train, two families dev, per intent.
        for rank, idx in enumerate(order):
            for text in families[idx]:
                row = {"text": text, "code": "I(" + intent + ")",
                       "intent": intent, "family_id": intent + ":" + str(idx)}
                (train if rank < 4 else dev).append(row)
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def audit_split(train, dev):
    train_families = {x["family_id"] for x in train}
    dev_families = {x["family_id"] for x in dev}
    if train_families & dev_families:
        raise ValueError("INTENT_FAMILY_LEAKAGE")
    if {x["text"] for x in train} & {x["text"] for x in dev}:
        raise ValueError("INTENT_SURFACE_LEAKAGE")
    if {x["intent"] for x in train} != {x["intent"] for x in dev}:
        raise ValueError("MISSING_INTENT_CLASS")
    return {"train_count": len(train), "dev_count": len(dev),
            "train_families": len(train_families),
            "dev_families": len(dev_families)}


if __name__ == "__main__":
    import json
    a, b = build_intent_examples()
    print(json.dumps(audit_split(a, b), ensure_ascii=False))

"""Text -> ordered sign entries (word-by-word glossing with fingerspelling fallback).

This is NOT translation into ASL grammar. Phase P4 adds an optional, clearly-labelled gloss normalisation step.
"""
import difflib

from .registry import Registry, SignEntry, norm

MAX_PHRASE = 4


def _entry(word: str, e: SignEntry) -> dict:
    return {"word": word, "gloss": e.gloss, "sign_id": e.id, "label": e.label, "kind": e.kind,
            "video_url": e.video["url"] if e.video else None, "fingerspelled": False, "letters": []}


def plan(text: str, reg: Registry) -> dict:
    words = norm(text).split()
    entries, missing, i = [], [], 0

    def find(phrase, fuzzy):
        if phrase in reg.by_label:
            return reg.by_label[phrase]
        if fuzzy and len(phrase) > 3:
            m = difflib.get_close_matches(phrase, [k for k in reg.by_label if len(k) > 1], 1, 0.85)
            return reg.by_label[m[0]] if m else None
        return None

    while i < len(words):
        for n in range(min(MAX_PHRASE, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            hit = find(phrase, fuzzy=(n == 1))
            if hit:
                entries.append(_entry(phrase, hit))
                i += n
                break
        else:
            word = words[i]
            letters = []
            for ch in word:
                e = reg.letter(ch)
                if e:
                    letters.append({"letter": ch.upper(), "sign_id": e.id,
                                    "video_url": e.video["url"] if e.video else None})
                elif ch.isalnum():
                    missing.append(ch)
            if letters:
                entries.append({"word": word, "gloss": "fs-" + word.upper(), "sign_id": None, "label": word,
                                "kind": "fingerspelling", "video_url": None, "fingerspelled": True,
                                "letters": letters})
            else:
                missing.append(word)
            i += 1
    return {"entries": entries, "missing": missing,
            "note": "Word-by-word glossing, not ASL grammar; real ASL word order and facial grammar differ."}

"""Text -> ordered sign entries (word-by-word glossing with fingerspelling fallback).

This is NOT translation into ASL grammar. With normalize=True an approximate English -> gloss step runs first:
articles, forms of "be" and infinitive "to" are dropped (ASL usually doesn't sign them), and inflected words are
matched to a known base sign (eating -> EAT, families -> FAMILY). Every change is reported in the result.
"""
import difflib

from .registry import Registry, SignEntry, norm

MAX_PHRASE = 4
DROP = {"a", "an", "the", "am", "is", "are", "was", "were", "be", "been", "being", "to"}
NOTE = "Word-by-word glossing, not ASL grammar; real ASL word order and facial grammar differ."


def _entry(word: str, e: SignEntry) -> dict:
    v = e.video or {}
    return {"word": word, "gloss": e.gloss, "sign_id": e.id, "label": e.label, "kind": e.kind,
            "video_url": v.get("url"), "attribution": v.get("attribution"), "license": v.get("license"),
            "fingerspelled": False, "letters": []}


def base_forms(word: str) -> list[str]:
    """Candidate base forms of an English word, most specific first (a light, rule-based lemmatiser)."""
    out = []
    if word.endswith("ies") and len(word) > 4:
        out.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        out.append(word[:-1])
    for suf in ("ing", "ed"):
        if word.endswith(suf) and len(word) > len(suf) + 2:
            stem = word[:-len(suf)]
            out += [stem, stem + "e"]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                out.append(stem[:-1])          # stopping -> stop
    if word.endswith("ied") and len(word) > 4:
        out.append(word[:-3] + "y")            # studied -> study
    return out


def plan(text: str, reg: Registry, normalize: bool = False) -> dict:
    words = norm(text).split()
    dropped = []
    if normalize:
        dropped = [w for w in words if w in DROP]
        words = [w for w in words if w not in DROP]
    entries, missing, i = [], [], 0

    def known(phrase):
        e = reg.by_label.get(phrase)
        return e if e and not (len(phrase) == 1 and e.category == "alphabet") else None

    def find(phrase, single):
        if known(phrase):
            return known(phrase), None
        if single and normalize:
            for b in base_forms(phrase):
                if known(b):
                    return known(b), phrase
        if single and len(phrase) > 3:
            m = difflib.get_close_matches(phrase, [k for k in reg.by_label if len(k) > 1], 1, 0.85)
            if m:
                return reg.by_label[m[0]], phrase
        return None, None

    while i < len(words):
        for n in range(min(MAX_PHRASE, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            hit, original = find(phrase, single=(n == 1))
            if hit:
                ent = _entry(phrase, hit)
                if original and norm(hit.label) != original:
                    ent["normalized_from"] = original
                entries.append(ent)
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
    note = NOTE
    if normalize:
        note += " Approximate gloss normalisation is on: dropped words and changed word forms are listed."
    return {"entries": entries, "missing": missing, "dropped": dropped, "normalized": normalize, "note": note}

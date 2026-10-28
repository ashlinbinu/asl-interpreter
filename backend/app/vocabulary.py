"""Tier-2 word vocabulary (docs/PLAN.md §4) and matching of dataset glosses to it. Pure: no DB access."""
import re
import string
from collections import defaultdict

WORDS = [
    "hello", "goodbye", "thank-you", "please", "sorry", "yes", "no", "help", "eat", "drink",
    "water", "mother", "father", "friend", "family", "love", "name", "good", "bad", "more",
    "finish", "want", "need", "where", "what", "who", "how", "why", "again", "stop",
    "go", "come", "home", "school", "work", "teacher", "book", "time", "like", "know",
    "learn", "understand", "happy", "sad", "tired", "hungry", "deaf", "hearing", "bathroom", "sign",
    "office",                       # added 2026-10-28 on request (ASL Citizen OFFICE: 30 clips, 28 signers)
]

# Other spellings of the same English word that a dataset may use. Only used when no exact gloss exists, and
# reported as such: a synonym gloss can be a different sign (e.g. TOILET vs BATHROOM), so check the video.
ALIASES = {"goodbye": ["BYE"], "bathroom": ["TOILET", "RESTROOM"], "hello": ["HI"]}

# Words are kept if a source has at least this many clips and distinct signers (PLAN.md §4).
MIN_CLIPS, MIN_SIGNERS = 15, 8


def gloss_of(word: str) -> str:
    return word.upper().replace(" ", "-")


def _key(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def gloss_keys(gloss: str) -> set[str]:
    """Letter-only keys a dataset gloss can match: 'THANK-YOU' -> THANKYOU, 'HELLO2' -> HELLO,
    'DONE (FINISH)' -> DONE and FINISH, 'WHO/WHOM' -> WHO and WHOM."""
    g = gloss.upper()
    parts = [re.sub(r"\(.*?\)", " ", g)] + re.findall(r"\((.*?)\)", g)
    keys = set()
    for p in parts:
        for q in [p] + p.split("/"):
            k = _key(q)
            if k:
                keys.add(k)
    return keys


_LETTER = re.compile(r"^(?:LETTER|FS|FINGERSPELL(?:ED)?|ALPHABET)?([A-Z])$")


def letter_of(gloss: str) -> str | None:
    """'A', 'A1', 'LETTER-A', 'FS-A', '#A' -> 'A'. Note that a one-letter gloss can be a word sign (e.g. pronoun I)."""
    k = _key(re.sub(r"\d", "", gloss))
    m = _LETTER.match(k)
    return m.group(1) if m else None


def summarize(rows) -> dict[str, dict]:
    """rows: iterable of objects with .gloss, .user, .split, .size -> per-gloss counts."""
    acc = defaultdict(lambda: {"clips": 0, "signers": set(), "bytes": 0,
                               "split": defaultdict(lambda: {"clips": 0, "signers": set()})})
    for r in rows:
        a = acc[r.gloss]
        a["clips"] += 1
        a["signers"].add(r.user)
        a["bytes"] += r.size or 0
        sp = a["split"][r.split]
        sp["clips"] += 1
        sp["signers"].add(r.user)
    out = {}
    for g, a in acc.items():
        out[g] = {"gloss": g, "clips": a["clips"], "signers": len(a["signers"]), "bytes": a["bytes"],
                  "split": {k: {"clips": v["clips"], "signers": len(v["signers"])}
                            for k, v in sorted(a["split"].items())}}
    return out


def match_words(stats: dict[str, dict], words=WORDS) -> dict[str, dict]:
    """For each word: candidate glosses (exact key first, then aliases) and the chosen one (most clips)."""
    by_key = defaultdict(list)
    for g in stats:
        for k in gloss_keys(g):
            by_key[k].append(g)
    out = {}
    for w in words:
        exact = sorted(set(by_key.get(_key(w), [])), key=lambda g: (-stats[g]["clips"], g))
        alias = sorted({g for a in ALIASES.get(w, []) for g in by_key.get(_key(a), [])} - set(exact),
                       key=lambda g: (-stats[g]["clips"], g))
        chosen = exact[0] if exact else (alias[0] if alias else None)
        s = stats.get(chosen)
        if s is None:
            status = "missing"
        elif s["clips"] >= MIN_CLIPS and s["signers"] >= MIN_SIGNERS:
            status = "ok" if chosen in exact else "alias"
        else:
            status = "low"
        out[w] = {"word": w, "gloss": gloss_of(w), "chosen": chosen, "status": status,
                  "exact": exact, "aliases": alias}
    return out


def match_letters(stats: dict[str, dict]) -> dict[str, list[str]]:
    """Letter -> dataset glosses that look like that letter."""
    out = {ch: [] for ch in string.ascii_uppercase}
    for g in sorted(stats):
        ch = letter_of(g)
        if ch:
            out[ch].append(g)
    return out

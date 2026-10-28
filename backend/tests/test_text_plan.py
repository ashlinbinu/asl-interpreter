"""Text -> sign plan: phrase matching, fingerspelling fallback, approximate gloss normalisation (flagged)."""
from app.registry import Registry, SignEntry, norm
from app.text_plan import base_forms, plan


def _reg(labels):
    r = Registry()
    entries = [SignEntry(i, lab, lab.upper().replace(" ", "-"), "word", "dynamic", None) for i, lab in enumerate(labels)]
    entries += [SignEntry(100 + k, ch, ch, "alphabet", "static", None) for k, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")]
    r.by_id = {e.id: e for e in entries}
    r.by_label = {norm(e.label): e for e in entries}
    return r


REG = _reg(["thank you", "eat", "family", "stop", "learn", "love", "mother", "home", "go"])


def test_base_forms():
    assert "eat" in base_forms("eating")
    assert "family" in base_forms("families")
    assert "stop" in base_forms("stopping") and "stop" in base_forms("stopped")
    assert "love" in base_forms("loved") and "learn" in base_forms("learned")


def test_plain_plan_keeps_every_word():
    r = plan("Thank you, the mother", REG)
    assert [e["gloss"] for e in r["entries"]] == ["THANK-YOU", "fs-THE", "MOTHER"]
    assert r["dropped"] == [] and not r["normalized"]


def test_normalised_plan_drops_function_words_and_lemmatises():
    r = plan("I am eating at the home of my families", REG, normalize=True)
    glosses = [e["gloss"] for e in r["entries"]]
    assert glosses == ["fs-I", "EAT", "fs-AT", "HOME", "fs-OF", "fs-MY", "FAMILY"]
    assert r["dropped"] == ["am", "the"]
    eat = next(e for e in r["entries"] if e["gloss"] == "EAT")
    assert eat["normalized_from"] == "eating" and "Approximate" in r["note"]
    assert plan("to go", REG, normalize=True)["dropped"] == ["to"]


def test_single_letters_are_fingerspelled_not_word_signs():
    r = plan("a b", REG)
    assert all(e["fingerspelled"] for e in r["entries"])

"""Learner-facing text: handshape descriptions and how to tell confusable letters apart."""

LETTERS = {
    "A": "Make a fist with the thumb resting against the side of the index finger, pointing up.",
    "B": "Hold four fingers straight up and together; fold the thumb across the palm.",
    "C": "Curve the fingers and thumb into a C shape, as if holding a cup.",
    "D": "Point the index finger up; curve the other fingers down to touch the thumb tip.",
    "E": "Bend all fingertips down over the thumb, which is tucked across the palm.",
    "F": "Touch the index fingertip to the thumb tip; keep the other three fingers up and apart.",
    "G": "Point the index finger and thumb sideways, parallel to each other; palm faces you.",
    "H": "Extend the index and middle fingers together, pointing sideways; palm faces you.",
    "I": "Raise the pinky; close the other fingers with the thumb across them.",
    "J": "Start in I and trace a J with the pinky (down, then curve toward you).",
    "K": "Index up, middle finger angled forward, thumb tip touching the middle finger between them.",
    "L": "Index finger up and thumb out to the side, forming an L.",
    "M": "Fold the index, middle and ring fingers over the thumb (thumb tip peeks out under the ring finger).",
    "N": "Fold the index and middle fingers over the thumb (thumb tip peeks out under the middle finger).",
    "O": "Curve all fingertips to meet the thumb tip, forming an O.",
    "P": "Make a K and point it down: middle finger toward the floor, wrist bent.",
    "Q": "Make a G and point it down: index finger and thumb toward the floor.",
    "R": "Cross the middle finger over the index finger; other fingers closed.",
    "S": "Make a fist with the thumb across the front of the fingers.",
    "T": "Tuck the thumb between the index and middle fingers, under the index finger.",
    "U": "Hold the index and middle fingers straight up and together.",
    "V": "Hold the index and middle fingers up and apart, in a V.",
    "W": "Hold the index, middle and ring fingers up and apart.",
    "X": "Raise the index finger and hook it; other fingers closed.",
    "Y": "Extend the thumb and pinky; close the other fingers.",
    "Z": "With the index finger, trace a Z in the air.",
}

DYNAMIC_LETTERS = {"J", "Z"}

# (target, looks_like) -> what to change
CONFUSIONS = {
    ("A", "S"): "Move your thumb to the side of your index finger, not across the front.",
    ("S", "A"): "Bring your thumb across the front of your fingers.",
    ("A", "E"): "Close the fist fully; for A the fingertips press into the palm, not onto the thumb.",
    ("E", "A"): "Bend the fingertips down so they rest on the thumb; tuck the thumb across the palm.",
    ("E", "S"): "Rest the fingertips on top of the thumb instead of wrapping the thumb over the fingers.",
    ("S", "E"): "Wrap the thumb over the front of the closed fingers.",
    ("M", "N"): "Fold three fingers (index, middle, ring) over the thumb, not two.",
    ("N", "M"): "Fold only two fingers (index and middle) over the thumb.",
    ("T", "N"): "Tuck the thumb between the index and middle fingers, under the index finger only.",
    ("N", "T"): "Lay the index and middle fingers over the thumb; the thumb shows under the middle finger.",
    ("M", "T"): "Fold three fingers over the thumb; the thumb tip shows under the ring finger.",
    ("T", "M"): "Only the index finger covers the thumb for T.",
    ("U", "V"): "Keep the two fingers pressed together.",
    ("V", "U"): "Spread the index and middle fingers apart.",
    ("U", "R"): "Keep the fingers side by side, not crossed.",
    ("R", "U"): "Cross the middle finger over the index finger.",
    ("V", "R"): "Spread the fingers apart instead of crossing them.",
    ("R", "V"): "Cross the fingers instead of spreading them.",
    ("K", "P"): "Point the hand up for K; P is the same shape pointing down.",
    ("P", "K"): "Bend the wrist so the middle finger points down.",
    ("G", "Q"): "Point the index finger sideways for G; Q points down.",
    ("Q", "G"): "Turn the hand so the index finger and thumb point down.",
    ("G", "H"): "Use only the index finger with the thumb; H uses two fingers.",
    ("H", "G"): "Add the middle finger next to the index finger.",
    ("H", "U"): "Turn the fingers sideways with the palm facing you.",
    ("U", "H"): "Point the fingers up, not sideways.",
    ("D", "X"): "Keep the index finger straight up and touch the other fingertips to the thumb.",
    ("X", "D"): "Hook the index finger; close the other fingers.",
    ("O", "C"): "Close the gap: fingertips should touch the thumb.",
    ("C", "O"): "Open a gap between the fingertips and the thumb.",
    ("F", "O"): "Raise the middle, ring and pinky fingers; only the index finger meets the thumb.",
    ("I", "Y"): "Tuck the thumb in; only the pinky is extended.",
    ("Y", "I"): "Stick the thumb out as well as the pinky.",
    ("L", "G"): "Point the index finger up, not sideways.",
    ("B", "U"): "Raise all four fingers.",
    ("W", "V"): "Add the ring finger: three fingers up.",
    ("V", "W"): "Fold the ring finger down: only two fingers up.",
}

CONFUSABLE_SETS = [
    ("A", "E", "M", "N", "S", "T"), ("U", "V", "R"), ("K", "P"), ("G", "Q"), ("G", "H"), ("I", "J"), ("D", "Z"),
]


def letter_hint(target: str, predicted: str | None) -> list[str]:
    """Specific hand-shape hints for a fingerspelling attempt."""
    target = target.upper()
    hints = []
    if target in DYNAMIC_LETTERS:
        hints.append(f"{target} is a moving letter: {LETTERS[target]}")
    elif predicted and predicted != target:
        hints.append(f"That looks more like {predicted}. " +
                     CONFUSIONS.get((target, predicted), f"Handshape for {target}: {LETTERS.get(target, '')}"))
    if not hints:
        hints.append(f"Handshape for {target}: {LETTERS.get(target, '')}")
    return hints

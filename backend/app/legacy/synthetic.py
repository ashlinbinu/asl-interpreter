"""DEV-ONLY legacy: procedurally generated hand templates. NOT real signing; never a source of truth.

Used only by scripts.seed_dev_synthetic when HANDSPEAK_DEV_SYNTHETIC=1.
"""
"""Canonical ASL hand landmark definitions for the full 26-letter alphabet (A-Z)
and common essential vocabulary.

Coordinates conform to MediaPipe's 21 3D hand landmark topology:
  0: Wrist
  1-4: Thumb (CMC, MCP, IP, Tip)
  5-8: Index (MCP, PIP, DIP, Tip)
  9-12: Middle (MCP, PIP, DIP, Tip)
  13-16: Ring (MCP, PIP, DIP, Tip)
  17-20: Pinky (MCP, PIP, DIP, Tip)

All output sequences are normalized to (32 frames, 126 features)
matching app.features.to_sequence:
  - First 63 floats: primary hand (21 x 3)
  - Last 63 floats: secondary hand (21 x 3, or zeros for one-handed signs)
"""
import numpy as np

T = 32
DIM = 126  # 2 hands x 21 landmarks x 3


def _build_finger(mcp, direction, length=0.45, curl=0.0):
    """
    Constructs 4 joint positions (MCP, PIP, DIP, TIP) for a finger.
    curl=0.0: extended straight out in `direction`
    curl=1.0: curled inward towards palm
    """
    mcp = np.array(mcp, dtype=float)
    d = np.array(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-8)

    seg = length / 3.0
    if curl <= 0.05:
        # Fully extended
        pip = mcp + d * seg
        dip = pip + d * seg
        tip = dip + d * seg
    elif curl >= 0.8:
        # Closed tight fist curl (bends in towards wrist/palm)
        down = np.array([0.0, 0.5, -0.4])
        down = down / np.linalg.norm(down)
        pip = mcp + (d * 0.4 + down * 0.6) * seg
        dip = pip + np.array([0.0, 0.4, 0.3]) * seg
        tip = dip + np.array([0.0, -0.2, 0.4]) * (seg * 0.8)
    else:
        # Partial curl / curve (e.g. C or O handshape)
        curve_down = np.array([0.0, 0.3, -0.7])
        curve_down = curve_down / np.linalg.norm(curve_down)
        pip = mcp + (d * 0.7 + curve_down * 0.3) * seg
        dip = pip + (d * 0.3 + curve_down * 0.7) * seg
        tip = dip + curve_down * seg

    return mcp, pip, dip, tip


def _make_hand(thumb_pos="side", index_curl=0.0, middle_curl=0.0, ring_curl=0.0, pinky_curl=0.0,
               spread=1.0, special=None):
    """Generates 21 keypoint coordinates for a right hand."""
    pts = np.zeros((21, 3))
    pts[0] = [0.0, 0.0, 0.0]  # Wrist at origin

    # Base MCP anchors
    mcp_thumb = np.array([-0.18, -0.25, -0.05])
    mcp_index = np.array([-0.14, -0.48, -0.05])
    mcp_middle = np.array([0.00, -0.50, -0.05])
    mcp_ring = np.array([0.12, -0.47, -0.05])
    mcp_pinky = np.array([0.22, -0.42, -0.05])

    # Direction vectors for extended fingers
    up = np.array([0.0, -1.0, 0.0])
    dir_index = np.array([-0.15 * spread, -0.98, -0.05])
    dir_middle = np.array([0.0, -1.0, -0.05])
    dir_ring = np.array([0.12 * spread, -0.98, -0.05])
    dir_pinky = np.array([0.25 * spread, -0.96, -0.05])

    # Build fingers
    pts[5], pts[6], pts[7], pts[8] = _build_finger(mcp_index, dir_index, 0.44, index_curl)
    pts[9], pts[10], pts[11], pts[12] = _build_finger(mcp_middle, dir_middle, 0.48, middle_curl)
    pts[13], pts[14], pts[15], pts[16] = _build_finger(mcp_ring, dir_ring, 0.43, ring_curl)
    pts[17], pts[18], pts[19], pts[20] = _build_finger(mcp_pinky, dir_pinky, 0.38, pinky_curl)

    # Thumb CMC & MCP
    pts[1] = mcp_thumb * 0.5
    pts[2] = mcp_thumb

    # Thumb position variations
    if thumb_pos == "up":  # Like in 'A'
        pts[3] = mcp_thumb + np.array([0.05, -0.22, 0.05])
        pts[4] = pts[3] + np.array([0.02, -0.20, 0.05])
    elif thumb_pos == "across":  # Folded over palm, like in 'B'
        pts[3] = mcp_thumb + np.array([0.18, -0.05, -0.08])
        pts[4] = pts[3] + np.array([0.15, 0.02, -0.08])
    elif thumb_pos == "out":  # L or Y thumb extended out 90 degrees
        pts[3] = mcp_thumb + np.array([-0.25, -0.15, -0.02])
        pts[4] = pts[3] + np.array([-0.22, -0.12, -0.02])
    elif thumb_pos == "touch_index":  # 'F' or 'D'
        pts[3] = pts[6] + np.array([-0.05, 0.05, -0.1])
        pts[4] = pts[8] + np.array([-0.02, 0.02, -0.05])
    elif thumb_pos == "touch_fingers":  # 'O'
        pts[3] = pts[10] + np.array([-0.05, 0.05, -0.15])
        pts[4] = (pts[8] + pts[12]) / 2.0 + np.array([0.0, 0.02, -0.05])
    elif thumb_pos == "tuck_middle":  # 'T'
        pts[3] = pts[5] + np.array([0.05, -0.10, -0.08])
        pts[4] = pts[3] + np.array([0.02, -0.12, -0.02])
    elif thumb_pos == "tuck_ring":  # 'N'
        pts[3] = pts[9] + np.array([0.08, -0.05, -0.08])
        pts[4] = pts[3] + np.array([0.02, -0.10, -0.02])
    elif thumb_pos == "tuck_pinky":  # 'M'
        pts[3] = pts[13] + np.array([0.08, -0.05, -0.08])
        pts[4] = pts[3] + np.array([0.02, -0.10, -0.02])
    elif thumb_pos == "front":  # 'S'
        pts[3] = mcp_middle + np.array([-0.05, 0.05, -0.15])
        pts[4] = pts[3] + np.array([0.15, 0.02, -0.05])
    else:  # default side
        pts[3] = mcp_thumb + np.array([-0.08, -0.18, -0.05])
        pts[4] = pts[3] + np.array([-0.05, -0.16, -0.05])

    # Special adjustments for specific letters
    if special == "C":
        # Curve all fingers and thumb into a C arc
        for f_base in [5, 9, 13, 17]:
            pts[f_base + 1][2] -= 0.15
            pts[f_base + 2][2] -= 0.25
            pts[f_base + 3][2] -= 0.22
            pts[f_base + 3][1] += 0.10
        pts[3] = np.array([-0.20, -0.15, -0.20])
        pts[4] = np.array([-0.18, -0.30, -0.22])
    elif special == "R":
        # Middle finger crosses over index finger
        pts[10] = pts[6] + np.array([-0.05, -0.02, 0.05])
        pts[11] = pts[7] + np.array([-0.06, -0.02, 0.05])
        pts[12] = pts[8] + np.array([-0.07, -0.02, 0.05])
    elif special == "X":
        # Index finger hooked
        pts[6] = pts[5] + np.array([0.0, -0.20, 0.0])
        pts[7] = pts[6] + np.array([0.0, -0.10, -0.15])
        pts[8] = pts[7] + np.array([0.0, 0.05, -0.12])
    elif special == "K":
        # Index up, middle forward at 45 deg, thumb resting in between
        pts[10] = pts[9] + np.array([0.0, -0.18, -0.15])
        pts[11] = pts[10] + np.array([0.0, -0.16, -0.15])
        pts[12] = pts[11] + np.array([0.0, -0.14, -0.12])
        pts[3] = (pts[5] + pts[9]) / 2.0 + np.array([0.0, -0.12, -0.08])
        pts[4] = pts[3] + np.array([0.0, -0.15, -0.05])
    elif special == "P":
        # K pointing down
        for i in range(1, 21):
            pts[i][1] = -pts[i][1] * 0.7
            pts[i][0] = pts[i][0] * 0.8
    elif special == "Q":
        # G pointing down
        for i in range(1, 21):
            pts[i][1] = -pts[i][1] * 0.7

    # Center wrist at (0, 0, 0) and scale hand
    pts = pts - pts[0]
    scale = np.linalg.norm(pts, axis=1).max() or 1.0
    return pts / scale


def _make_sequence(base_hand, dynamic_motion=None):
    """
    Takes 21x3 base hand pose and produces (32, 126) sequence.
    Subtle organic micro-motion (breathing / stabilization) prevents completely rigid skeletons.
    dynamic_motion: optional function(t: float) -> (offset_x, offset_y, offset_z)
    """
    seq = np.zeros((T, DIM))
    for t_idx in range(T):
        t = t_idx / (T - 1.0)
        h = base_hand.copy()

        # Natural micro-motion
        wobble = 0.015 * np.sin(t * 2 * np.pi)
        h[:, 0] += wobble * 0.5
        h[:, 1] += wobble * 0.3

        if dynamic_motion:
            dx, dy, dz = dynamic_motion(t)
            h[:, 0] += dx
            h[:, 1] += dy
            h[:, 2] += dz

        # Re-normalize
        h = h - h[0]
        scale = np.linalg.norm(h, axis=1).max() or 1.0
        norm_h = h / scale

        # Primary hand in first 63 numbers
        seq[t_idx, :63] = norm_h.reshape(-1)

    return seq


def _get_letter_pose(char):
    """Constructs the canonical 21-landmark base pose for ASL alphabet letters."""
    c = char.upper()
    if c == "A":
        return _make_hand(thumb_pos="up", index_curl=1.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
    elif c == "B":
        return _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=0.0, ring_curl=0.0, pinky_curl=0.0, spread=0.2)
    elif c == "C":
        return _make_hand(thumb_pos="side", index_curl=0.5, middle_curl=0.5, ring_curl=0.5, pinky_curl=0.5, special="C")
    elif c == "D":
        return _make_hand(thumb_pos="touch_index", index_curl=0.0, middle_curl=0.8, ring_curl=0.8, pinky_curl=0.8)
    elif c == "E":
        return _make_hand(thumb_pos="across", index_curl=0.9, middle_curl=0.9, ring_curl=0.9, pinky_curl=0.9)
    elif c == "F":
        return _make_hand(thumb_pos="touch_index", index_curl=0.8, middle_curl=0.0, ring_curl=0.0, pinky_curl=0.0, spread=1.2)
    elif c == "G":
        # Index pointing horizontally forward, thumb parallel
        h = _make_hand(thumb_pos="side", index_curl=0.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
        h[5:9, 0] += 0.25
        h[5:9, 1] += 0.3
        return h
    elif c == "H":
        # Index & middle pointing horizontally forward
        h = _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=0.0, ring_curl=1.0, pinky_curl=1.0, spread=0.1)
        h[5:13, 0] += 0.25
        h[5:13, 1] += 0.3
        return h
    elif c == "I":
        return _make_hand(thumb_pos="across", index_curl=1.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=0.0)
    elif c == "J":
        # I handshape that traces a J curve over time
        return _make_hand(thumb_pos="across", index_curl=1.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=0.0)
    elif c == "K":
        return _make_hand(thumb_pos="side", index_curl=0.0, middle_curl=0.0, ring_curl=1.0, pinky_curl=1.0, special="K")
    elif c == "L":
        return _make_hand(thumb_pos="out", index_curl=0.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
    elif c == "M":
        return _make_hand(thumb_pos="tuck_pinky", index_curl=0.9, middle_curl=0.9, ring_curl=0.9, pinky_curl=1.0)
    elif c == "N":
        return _make_hand(thumb_pos="tuck_ring", index_curl=0.9, middle_curl=0.9, ring_curl=1.0, pinky_curl=1.0)
    elif c == "O":
        return _make_hand(thumb_pos="touch_fingers", index_curl=0.6, middle_curl=0.6, ring_curl=0.6, pinky_curl=0.6)
    elif c == "P":
        return _make_hand(thumb_pos="side", index_curl=0.0, middle_curl=0.0, ring_curl=1.0, pinky_curl=1.0, special="P")
    elif c == "Q":
        return _make_hand(thumb_pos="side", index_curl=0.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0, special="Q")
    elif c == "R":
        return _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=0.0, ring_curl=1.0, pinky_curl=1.0, special="R")
    elif c == "S":
        return _make_hand(thumb_pos="front", index_curl=1.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
    elif c == "T":
        return _make_hand(thumb_pos="tuck_middle", index_curl=0.9, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
    elif c == "U":
        return _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=0.0, ring_curl=1.0, pinky_curl=1.0, spread=0.1)
    elif c == "V":
        return _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=0.0, ring_curl=1.0, pinky_curl=1.0, spread=1.2)
    elif c == "W":
        return _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=0.0, ring_curl=0.0, pinky_curl=1.0, spread=1.2)
    elif c == "X":
        return _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0, special="X")
    elif c == "Y":
        return _make_hand(thumb_pos="out", index_curl=1.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=0.0)
    elif c == "Z":
        # Index finger up that traces a Z path over time
        return _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
    else:
        return _make_hand(thumb_pos="up", index_curl=0.0, middle_curl=0.0, ring_curl=0.0, pinky_curl=0.0)


def get_alphabet_sequence(char):
    """Returns (32, 126) sequence for an ASL letter."""
    c = char.upper()
    base = _get_letter_pose(c)

    # Dynamic trajectories for letters J and Z
    motion = None
    if c == "J":
        def j_motion(t):
            # Curves down and hooks up to the left
            if t < 0.6:
                return (0.0, t * 0.3, 0.0)
            else:
                progress = (t - 0.6) / 0.4
                theta = progress * np.pi
                return (-np.sin(theta) * 0.15, 0.18 - np.cos(theta) * 0.08, 0.0)
        motion = j_motion
    elif c == "Z":
        def z_motion(t):
            # Top stroke (left to right), diagonal stroke, bottom stroke
            if t < 0.33:
                return ((t / 0.33) * 0.2 - 0.1, -0.1, 0.0)
            elif t < 0.66:
                p = (t - 0.33) / 0.33
                return (0.1 - p * 0.2, -0.1 + p * 0.2, 0.0)
            else:
                p = (t - 0.66) / 0.34
                return (-0.1 + p * 0.2, 0.1, 0.0)
        motion = z_motion

    return _make_sequence(base, motion)


def get_common_phrases():
    """Generates standard dynamic multi-frame gestures for common phrases."""
    phrases = {}

    # 1. Hello (Open hand waving at temple)
    open_hand = _make_hand(thumb_pos="out", index_curl=0.0, middle_curl=0.0, ring_curl=0.0, pinky_curl=0.0, spread=0.5)
    def hello_motion(t):
        return (0.2 + np.sin(t * 3 * np.pi) * 0.15, -0.2 + t * 0.1, 0.0)
    phrases["hello"] = _make_sequence(open_hand, hello_motion)

    # 2. Thank you (Hand from chin moving forward and down)
    flat_hand = _make_hand(thumb_pos="across", index_curl=0.0, middle_curl=0.0, ring_curl=0.0, pinky_curl=0.0, spread=0.2)
    def thanks_motion(t):
        return (0.0, -0.2 + t * 0.3, -t * 0.4)
    phrases["thank you"] = _make_sequence(flat_hand, thanks_motion)

    # 3. Yes (S-fist nodding up and down)
    fist = _make_hand(thumb_pos="front", index_curl=1.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
    def yes_motion(t):
        return (0.0, np.sin(t * 4 * np.pi) * 0.12, 0.0)
    phrases["yes"] = _make_sequence(fist, yes_motion)

    # 4. No (Index and middle snapping down to thumb)
    def no_motion(t):
        # Open and snap shut
        return (0.0, np.sin(t * 3 * np.pi) * 0.08, 0.0)
    phrases["no"] = _make_sequence(_get_letter_pose("N"), no_motion)

    # 5. Please (Open hand circling on chest)
    def please_motion(t):
        theta = t * 4 * np.pi
        return (np.cos(theta) * 0.12, np.sin(theta) * 0.12, 0.0)
    phrases["please"] = _make_sequence(flat_hand, please_motion)

    # 6. Help (Thumbs-up fist resting on flat palm lifting up)
    thumbs_up = _make_hand(thumb_pos="up", index_curl=1.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=1.0)
    def help_motion(t):
        return (0.0, -t * 0.25, 0.0)
    phrases["help"] = _make_sequence(thumbs_up, help_motion)

    # 7. Love (I Love You handshape: thumb, index, pinky out)
    ily = _make_hand(thumb_pos="out", index_curl=0.0, middle_curl=1.0, ring_curl=1.0, pinky_curl=0.0, spread=1.2)
    phrases["love"] = _make_sequence(ily)

    # 8. Goodbye (Open hand waving fingers up and down)
    def bye_motion(t):
        return (0.1, -0.15 + np.sin(t * 4 * np.pi) * 0.1, 0.0)
    phrases["goodbye"] = _make_sequence(open_hand, bye_motion)

    # 9. Water (W handshape tapping chin)
    w_hand = _get_letter_pose("W")
    def water_motion(t):
        return (0.0, np.sin(t * 4 * np.pi) * 0.06, 0.0)
    phrases["water"] = _make_sequence(w_hand, water_motion)

    # 10. Friend (X fingers linking)
    x_hand = _get_letter_pose("X")
    def friend_motion(t):
        return (np.sin(t * 2 * np.pi) * 0.08, np.cos(t * 2 * np.pi) * 0.05, 0.0)
    phrases["friend"] = _make_sequence(x_hand, friend_motion)

    return phrases


def get_all_canonical_signs():
    """Returns list of dicts for all alphabet letters (A-Z) and common vocabulary."""
    all_signs = []

    # 26 Alphabet letters
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        seq = get_alphabet_sequence(char)
        all_signs.append({
            "label": char,
            "frames": seq.tolist(),
            "source": "alphabet",
            "video_url": None,
        })

    # Common vocabulary phrases
    for label, seq in get_common_phrases().items():
        all_signs.append({
            "label": label,
            "frames": seq.tolist(),
            "source": "common",
            "video_url": None,
        })

    return all_signs

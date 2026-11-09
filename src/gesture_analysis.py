"""Transparent geometric finger extension and gesture rules."""
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]

def count_extended_fingers(landmarks):
    if landmarks is None or len(landmarks) < 21:
        return {"count": None, "extended": [], "states": {}}
    # Image y grows downward. For four fingers, tip above the middle joint indicates extension.
    states = {name: landmarks[tip][1] < landmarks[pip][1] for name, tip, pip in zip(FINGER_NAMES[1:], TIPS[1:], PIPS[1:])}
    # Thumb extension is approximated by distance from wrist relative to its IP joint.
    states["thumb"] = abs(landmarks[4][0]-landmarks[0][0]) > abs(landmarks[3][0]-landmarks[0][0]) * 1.15
    extended = [name for name in FINGER_NAMES if states.get(name, False)]
    return {"count": len(extended), "extended": extended, "states": states}

def recognize_gesture(landmarks):
    result = count_extended_fingers(landmarks)
    if result["count"] is None: return "Unknown"
    s = result["states"]
    if s["thumb"] and not any(s[n] for n in FINGER_NAMES[1:]): return "Thumbs up" if landmarks[4][1] < landmarks[3][1] else "Thumbs up (approx.)"
    if result["count"] == 0: return "Fist"
    if result["count"] == 5: return "Open palm"
    if s["index"] and s["middle"] and not s["ring"] and not s["little"]: return "Peace sign"
    if s["index"] and not any(s[n] for n in ("middle", "ring", "little")): return "One finger raised"
    if result["count"] == 2: return "Two fingers raised"
    return "Other"

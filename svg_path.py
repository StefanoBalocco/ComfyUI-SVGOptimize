"""Exact parsing and shortest-text writing of SVG path data."""
import re
from decimal import Decimal, Inexact

from .svg_numbers import EXACT, NUMBER, SEPARATOR, SPACE, format_number, join_numbers, separator

ARGUMENTS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
COMMANDS = "MmLlHhVvCcSsQqTtAaZz"
FLAG = re.compile(r"[01]")


def tokenize(d):
    """Split path data into (letter, values) groups, one group per argument set."""
    groups = []
    pos = SPACE.match(d).end()
    while pos < len(d):
        letter = d[pos]
        if letter not in COMMANDS:
            return None
        count = ARGUMENTS[letter.upper()]
        pos = SPACE.match(d, pos + 1).end()
        if count == 0:
            groups.append((letter, ()))
            continue
        while True:
            values = []
            for index in range(count):
                if index:
                    pos = SEPARATOR.match(d, pos).end()
                pattern = FLAG if letter in "Aa" and index in (3, 4) else NUMBER
                match = pattern.match(d, pos)
                if match is None:
                    return None
                values.append(Decimal(match.group()))
                pos = match.end()
            groups.append((letter, tuple(values)))
            if letter in "Mm":
                letter = "L" if letter == "M" else "l"
            following = SEPARATOR.match(d, pos).end()
            if NUMBER.match(d, following) is None:
                break
            pos = following
        pos = SPACE.match(d, pos).end()
    return groups


def shift(command, values, dx, dy):
    """Add dx to every x and dy to every y; an arc keeps its radii, rotation and flags."""
    head = values[:5] if command == "A" else ()
    tail = values[len(head):]
    return head + tuple(EXACT.add(v, dy if index % 2 else dx) for index, v in enumerate(tail))


def reflected_control(previous, command, x, y):
    """First control point of an S or T segment (written as C or Q) after the previous segment."""
    if previous is not None and previous[0] == command:
        return EXACT.subtract(EXACT.add(x, x), previous[1][-4]), EXACT.subtract(EXACT.add(y, y), previous[1][-3])
    return x, y


def parse_path(d):
    groups = tokenize(d)
    if groups is None:
        return None
    if groups and groups[0][0] not in "Mm":
        return None
    segments = []
    x = y = start_x = start_y = Decimal(0)
    try:
        for letter, values in groups:
            command = letter.upper()
            previous = segments[-1] if segments else None
            if command == "H":
                values = (EXACT.add(x, values[0]) if letter == "h" else values[0], y)
                command = "L"
            elif command == "V":
                values = (x, EXACT.add(y, values[0]) if letter == "v" else values[0])
                command = "L"
            elif letter != command:
                values = shift(command, values, x, y)
            if command == "S":
                values = reflected_control(previous, "C", x, y) + values
                command = "C"
            elif command == "T":
                values = reflected_control(previous, "Q", x, y) + values
                command = "Q"
            if command == "Z":
                x, y = start_x, start_y
            else:
                x, y = values[-2:]
                if command == "M":
                    start_x, start_y = x, y
            segments.append((command, values))
    except Inexact:
        return None
    return segments


def format_path(segments):
    out = []
    letter = last = ""
    x = y = start_x = start_y = Decimal(0)
    try:
        for index, (command, values) in enumerate(segments):
            if command == "Z":
                out.append("z")
                letter, last = "z", ""
                x, y = start_x, start_y
                continue
            candidates = [(command, values)]
            if index:
                relative = shift(command, values, EXACT.minus(x), EXACT.minus(y))
                relative_short = []
                absolute_short = []
                previous = segments[index - 1]
                if command == "L":
                    if relative[1] == 0:
                        relative_short.append(("h", relative[:1]))
                        absolute_short.append(("H", values[:1]))
                    if relative[0] == 0:
                        relative_short.append(("v", relative[1:]))
                        absolute_short.append(("V", values[1:]))
                elif command in "CQ" and values[:2] == reflected_control(previous, command, x, y):
                    short = "t" if command == "Q" else "s"
                    relative_short.append((short, relative[2:]))
                    absolute_short.append((short.upper(), values[2:]))
                candidates = relative_short + absolute_short + [(command.lower(), relative), (command, values)]
            best = None
            for candidate, numbers in candidates:
                texts = [format_number(v) for v in numbers]
                if (candidate == letter and candidate not in "Mm") or (letter, candidate) in (("M", "L"), ("m", "l")):
                    text = separator(last, texts[0]) + join_numbers(texts)
                else:
                    text = candidate + join_numbers(texts)
                if best is None or len(text) < len(best[0]):
                    best = text, candidate, texts[-1]
            out.append(best[0])
            _, letter, last = best
            x, y = values[-2:]
            if command == "M":
                start_x, start_y = x, y
    except Inexact:
        return None
    return "".join(out)


def translate_path(segments, dx, dy):
    try:
        return [(command, shift(command, values, dx, dy)) for command, values in segments]
    except Inexact:
        return None

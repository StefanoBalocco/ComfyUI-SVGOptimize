"""Parsing of SVG transform lists and writing of shorter equivalent forms."""
import re
from decimal import Decimal

from .svg_numbers import NUMBER, SEPARATOR, SPACE, format_numbers

ARGUMENTS = {"matrix": (6,), "translate": (1, 2), "scale": (1, 2), "rotate": (1, 3), "skewX": (1,), "skewY": (1,)}
NAME = re.compile(r"[A-Za-z]+")


def parse_transform(text):
    """Return the (name, arguments) items of a transform list, or None when it is malformed."""
    items = []
    pos = SPACE.match(text).end()
    while pos < len(text):
        name = NAME.match(text, pos)
        if name is None or name.group() not in ARGUMENTS:
            return None
        pos = SPACE.match(text, name.end()).end()
        if text[pos:pos + 1] != "(":
            return None
        pos = SPACE.match(text, pos + 1).end()
        values = []
        while True:
            number = NUMBER.match(text, pos)
            if number is None:
                return None
            values.append(Decimal(number.group()))
            pos = number.end()
            following = SEPARATOR.match(text, pos).end()
            if NUMBER.match(text, following) is None:
                break
            pos = following
        pos = SPACE.match(text, pos).end()
        if text[pos:pos + 1] != ")" or len(values) not in ARGUMENTS[name.group()]:
            return None
        items.append((name.group(), tuple(values)))
        following = SEPARATOR.match(text, pos + 1).end()
        if following == len(text) and following > SPACE.match(text, pos + 1).end():
            return None
        pos = following
    return items


def shorten(name, values):
    if name == "matrix":
        a, b, c, d, e, f = values
        if b == c == e == f == 0:
            name, values = "scale", (a, d)
        elif (a, b, c, d) == (1, 0, 0, 1):
            name, values = "translate", (e, f)
    if name == "translate" and len(values) == 2 and values[1] == 0:
        values = values[:1]
    elif name == "scale" and len(values) == 2 and values[0] == values[1]:
        values = values[:1]
    elif name == "rotate" and len(values) == 3 and values[1] == values[2] == 0:
        values = values[:1]
    return name, values


def is_identity(name, values):
    if name == "scale":
        return all(v == 1 for v in values)
    if name == "matrix":
        return values == (1, 0, 0, 1, 0, 0)
    return values[0] == 0 and (name != "translate" or len(values) == 1)


def format_transform(items):
    """Write the list in its shortest form; identity functions are dropped."""
    parts = []
    for name, values in items:
        name, values = shorten(name, values)
        if not is_identity(name, values):
            parts.append(f"{name}({format_numbers(values)})")
    return " ".join(parts)


def as_translate(items):
    """Return (x, y) when the list is a single translate, else None."""
    if len(items) != 1:
        return None
    name, values = items[0]
    if name == "translate":
        return values[0], values[1] if len(values) == 2 else Decimal(0)
    if name == "matrix" and values[:4] == (1, 0, 0, 1):
        return values[4], values[5]
    return None

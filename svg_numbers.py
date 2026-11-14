"""Exact number parsing and shortest-text formatting for SVG numbers."""
import re
from decimal import Context, Decimal, Inexact, InvalidOperation

NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
SPACE = re.compile(r"[ \t\n\r\f]*")
SEPARATOR = re.compile(r"[ \t\n\r\f]*(?:,[ \t\n\r\f]*)?")
EXACT = Context(prec=1000, traps=[Inexact, InvalidOperation])


def parse_number(text):
    if NUMBER.fullmatch(text) is None:
        return None
    return Decimal(text)


def format_number(value):
    if value == 0:
        return "0"
    sign, digits, exponent = value.as_tuple()
    digits = "".join(map(str, digits))
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    digits = stripped
    sign = "-" if sign else ""
    if exponent >= 0:
        positional = len(digits) + exponent
    elif -exponent >= len(digits):
        positional = 1 - exponent
    else:
        positional = len(digits) + 1
    if len(digits) + 1 + len(str(exponent)) < positional:
        return f"{sign}{digits}e{exponent}"
    if exponent >= 0:
        return sign + digits + "0" * exponent
    if -exponent >= len(digits):
        return sign + "." + "0" * (-exponent - len(digits)) + digits
    return sign + digits[:exponent] + "." + digits[exponent:]


def separator(previous, following):
    if not previous or following.startswith("-"):
        return ""
    if following.startswith(".") and ("." in previous or "e" in previous):
        return ""
    return " "


def join_numbers(texts):
    out = []
    previous = ""
    for text in texts:
        out.append(separator(previous, text) + text)
        previous = text
    return "".join(out)


def format_numbers(values):
    return join_numbers([format_number(value) for value in values])

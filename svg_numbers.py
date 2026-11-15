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


def scientific(value):
    sign, digits, exponent = value.as_tuple()
    digits = "".join(map(str, digits))
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    return f"{'-' if sign else ''}{stripped}e{exponent}"


def format_number(value):
    if value == 0:
        return "0"
    if abs(value.adjusted()) > 100000:
        # compare lengths first so that huge plain text is only built when it is the shorter form
        short = scientific(value)
        digits, exponent = short.lstrip("-").split("e")
        digits, exponent = len(digits), int(exponent)
        if exponent >= 0:
            plain = digits + exponent
        elif -exponent >= digits:
            plain = 1 - exponent
        else:
            plain = digits + 1
        if len(short) < plain + (value < 0):
            return short
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    unsigned = text.lstrip("-")
    if (unsigned.endswith("000") and "." not in unsigned) or unsigned.startswith(".000"):
        short = scientific(value)
        if len(short) < len(text):
            return short
    return text


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

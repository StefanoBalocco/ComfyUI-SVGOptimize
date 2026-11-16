"""Shortest text for colors and lengths, style attribute parsing, and presentation attribute tables."""
import re
from decimal import Decimal, Inexact

from .svg_numbers import EXACT, NUMBER, format_number, parse_number

COLOR_KEYWORDS = {
    "aliceblue": "#f0f8ff", "antiquewhite": "#faebd7", "aqua": "#00ffff", "aquamarine": "#7fffd4",
    "azure": "#f0ffff", "beige": "#f5f5dc", "bisque": "#ffe4c4", "black": "#000000",
    "blanchedalmond": "#ffebcd", "blue": "#0000ff", "blueviolet": "#8a2be2", "brown": "#a52a2a",
    "burlywood": "#deb887", "cadetblue": "#5f9ea0", "chartreuse": "#7fff00", "chocolate": "#d2691e",
    "coral": "#ff7f50", "cornflowerblue": "#6495ed", "cornsilk": "#fff8dc", "crimson": "#dc143c",
    "cyan": "#00ffff", "darkblue": "#00008b", "darkcyan": "#008b8b", "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9", "darkgreen": "#006400", "darkgrey": "#a9a9a9", "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b", "darkolivegreen": "#556b2f", "darkorange": "#ff8c00", "darkorchid": "#9932cc",
    "darkred": "#8b0000", "darksalmon": "#e9967a", "darkseagreen": "#8fbc8f", "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f", "darkslategrey": "#2f4f4f", "darkturquoise": "#00ced1", "darkviolet": "#9400d3",
    "deeppink": "#ff1493", "deepskyblue": "#00bfff", "dimgray": "#696969", "dimgrey": "#696969",
    "dodgerblue": "#1e90ff", "firebrick": "#b22222", "floralwhite": "#fffaf0", "forestgreen": "#228b22",
    "fuchsia": "#ff00ff", "gainsboro": "#dcdcdc", "ghostwhite": "#f8f8ff", "gold": "#ffd700",
    "goldenrod": "#daa520", "gray": "#808080", "green": "#008000", "greenyellow": "#adff2f",
    "grey": "#808080", "honeydew": "#f0fff0", "hotpink": "#ff69b4", "indianred": "#cd5c5c",
    "indigo": "#4b0082", "ivory": "#fffff0", "khaki": "#f0e68c", "lavender": "#e6e6fa",
    "lavenderblush": "#fff0f5", "lawngreen": "#7cfc00", "lemonchiffon": "#fffacd", "lightblue": "#add8e6",
    "lightcoral": "#f08080", "lightcyan": "#e0ffff", "lightgoldenrodyellow": "#fafad2", "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90", "lightgrey": "#d3d3d3", "lightpink": "#ffb6c1", "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa", "lightskyblue": "#87cefa", "lightslategray": "#778899", "lightslategrey": "#778899",
    "lightsteelblue": "#b0c4de", "lightyellow": "#ffffe0", "lime": "#00ff00", "limegreen": "#32cd32",
    "linen": "#faf0e6", "magenta": "#ff00ff", "maroon": "#800000", "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd", "mediumorchid": "#ba55d3", "mediumpurple": "#9370db", "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee", "mediumspringgreen": "#00fa9a", "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585", "midnightblue": "#191970", "mintcream": "#f5fffa", "mistyrose": "#ffe4e1",
    "moccasin": "#ffe4b5", "navajowhite": "#ffdead", "navy": "#000080", "oldlace": "#fdf5e6",
    "olive": "#808000", "olivedrab": "#6b8e23", "orange": "#ffa500", "orangered": "#ff4500",
    "orchid": "#da70d6", "palegoldenrod": "#eee8aa", "palegreen": "#98fb98", "paleturquoise": "#afeeee",
    "palevioletred": "#db7093", "papayawhip": "#ffefd5", "peachpuff": "#ffdab9", "peru": "#cd853f",
    "pink": "#ffc0cb", "plum": "#dda0dd", "powderblue": "#b0e0e6", "purple": "#800080",
    "red": "#ff0000", "rosybrown": "#bc8f8f", "royalblue": "#4169e1", "saddlebrown": "#8b4513",
    "salmon": "#fa8072", "sandybrown": "#f4a460", "seagreen": "#2e8b57", "seashell": "#fff5ee",
    "sienna": "#a0522d", "silver": "#c0c0c0", "skyblue": "#87ceeb", "slateblue": "#6a5acd",
    "slategray": "#708090", "slategrey": "#708090", "snow": "#fffafa", "springgreen": "#00ff7f",
    "steelblue": "#4682b4", "tan": "#d2b48c", "teal": "#008080", "thistle": "#d8bfd8",
    "tomato": "#ff6347", "turquoise": "#40e0d0", "violet": "#ee82ee", "wheat": "#f5deb3",
    "white": "#ffffff", "whitesmoke": "#f5f5f5", "yellow": "#ffff00", "yellowgreen": "#9acd32",
}

COLOR_PROPERTIES = {"fill", "stroke", "stop-color", "flood-color", "lighting-color", "color"}
LENGTH_ATTRIBUTES = {"x", "y", "width", "height", "cx", "cy", "r", "rx", "ry", "x1", "y1", "x2", "y2",
                     "stroke-width", "stroke-miterlimit", "stroke-dashoffset", "opacity", "fill-opacity",
                     "stroke-opacity", "stop-opacity", "flood-opacity", "offset", "font-size"}

PRESENTATION_ATTRIBUTES = frozenset("""
    alignment-baseline baseline-shift clip clip-path clip-rule color color-interpolation
    color-interpolation-filters color-profile color-rendering cursor direction display dominant-baseline
    enable-background fill fill-opacity fill-rule filter flood-color flood-opacity font-family font-size
    font-size-adjust font-stretch font-style font-variant font-weight glyph-orientation-horizontal
    glyph-orientation-vertical image-rendering kerning letter-spacing lighting-color marker-end marker-mid
    marker-start mask opacity overflow pointer-events shape-rendering stop-color stop-opacity stroke
    stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity
    stroke-width text-anchor text-decoration text-rendering unicode-bidi visibility word-spacing writing-mode
""".split())

INHERITED_PROPERTIES = PRESENTATION_ATTRIBUTES - frozenset("""
    alignment-baseline baseline-shift clip clip-path display dominant-baseline enable-background filter
    flood-color flood-opacity lighting-color mask opacity overflow stop-color stop-opacity text-decoration
    unicode-bidi
""".split())

INHERITED_DEFAULTS = {
    "fill": "#000", "fill-opacity": "1", "fill-rule": "nonzero", "clip-rule": "nonzero", "stroke": "none",
    "stroke-width": "1", "stroke-opacity": "1", "stroke-linecap": "butt", "stroke-linejoin": "miter",
    "stroke-miterlimit": "4", "stroke-dasharray": "none", "stroke-dashoffset": "0", "visibility": "visible",
}

OTHER_DEFAULTS = {"opacity": "1", "stop-opacity": "1", "flood-opacity": "1", "stop-color": "#000",
                  "flood-color": "#000", "lighting-color": "#fff"}

GEOMETRY_DEFAULTS = {
    "svg": {"x": "0", "y": "0"},
    "rect": {"x": "0", "y": "0"}, "use": {"x": "0", "y": "0"}, "image": {"x": "0", "y": "0"},
    "circle": {"cx": "0", "cy": "0"}, "ellipse": {"cx": "0", "cy": "0"},
    "line": {"x1": "0", "y1": "0", "x2": "0", "y2": "0"},
}

HEX_COLOR = re.compile(r"#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})", re.IGNORECASE)
COLOR_FUNCTION = re.compile(r"rgba?\(([^()]*)\)", re.IGNORECASE)
CHANNEL_SEPARATOR = re.compile(r"[ \t\n\r\f]+")
LENGTH = re.compile(rf"({NUMBER.pattern})(px|%|[a-zA-Z]+)?")
ZERO_LENGTH = re.compile(r"[+-]?0*\.?0+(?:px|%|em|ex|in|cm|mm|pt|pc)?")
CSS_SPACE = " \t\r\n\f"
IMPORTANT = re.compile(r"![ \t\n\r\f]*important", re.IGNORECASE)

KEYWORDS_BY_HEX = {}
for keyword, hex_color in sorted(COLOR_KEYWORDS.items()):
    KEYWORDS_BY_HEX.setdefault(hex_color, []).append(keyword)


def percent_to_byte(text):
    percent = parse_number(text)
    if percent is None or not 0 <= percent <= 100:
        return None
    byte = EXACT.divide(EXACT.multiply(percent, 255), 100)
    return int(byte) if byte == byte.to_integral_value() else None


def alpha_in_255ths(text):
    """The alpha written as a number or percentage, times 255, or None when it is not within 0 to 1."""
    percent = text.endswith("%")
    alpha = parse_number(text[:-1] if percent else text)
    if alpha is None:
        return None
    alpha = EXACT.multiply(alpha, Decimal("2.55") if percent else 255)
    return alpha if 0 <= alpha <= 255 else None


def function_color(body):
    """The color of the arguments of rgb() or rgba(), written with commas or with spaces and a slash."""
    body = body.strip(CSS_SPACE)
    if "," in body:
        channels = [item.strip(CSS_SPACE) for item in body.split(",")]
        channels, alpha = channels[:3], channels[3:]
    else:
        channels, slash, alpha = body.partition("/")
        channels = CHANNEL_SEPARATOR.split(channels.strip(CSS_SPACE))
        alpha = [alpha.strip(CSS_SPACE)] if slash else []
    if len(channels) != 3 or len(alpha) > 1:
        return None
    if all(re.fullmatch("[0-9]+", channel) for channel in channels):
        values = tuple(int(channel) for channel in channels)
        if max(values) > 255:
            return None
    elif all(channel.endswith("%") for channel in channels):
        values = tuple(percent_to_byte(channel[:-1]) for channel in channels)
        if None in values:
            return None
    else:
        return None
    alpha = alpha_in_255ths(alpha[0]) if alpha else Decimal(255)
    return None if alpha is None else values + (alpha,)


def parse_color(text):
    """The red, green and blue bytes and the alpha times 255 of a color written as hex, keyword, `transparent` or rgb()/rgba(), or None."""
    if HEX_COLOR.fullmatch(text):
        digits = text[1:].lower()
        if len(digits) <= 4:
            digits = "".join(digit * 2 for digit in digits)
        values = tuple(int(digits[i:i + 2], 16) for i in range(0, len(digits), 2))
        return values[:3] + (Decimal(values[3] if len(values) == 4 else 255),)
    if text.lower() == "transparent":
        return 0, 0, 0, Decimal(0)
    if text.lower() in COLOR_KEYWORDS:
        return parse_color(COLOR_KEYWORDS[text.lower()])
    match = COLOR_FUNCTION.fullmatch(text)
    if match is None:
        return None
    try:
        return function_color(match.group(1))
    except Inexact:
        return None


def hsl_text(rgb, opacity):
    """The color as hsl() text, or None when its numbers are not exact decimals."""
    red, green, blue = rgb
    high, low = max(rgb), min(rgb)
    delta = high - low
    try:
        lightness = EXACT.divide(10 * (high + low), 51)
        if delta == 0:
            hue = saturation = Decimal(0)
        else:
            saturation = EXACT.divide(100 * delta, 255 - abs(high + low - 255))
            if high == red:
                steps = green - blue
            elif high == green:
                steps = blue - red + 2 * delta
            else:
                steps = red - green + 4 * delta
            hue = EXACT.divide(60 * (steps % (6 * delta)), delta)
    except Inexact:
        return None
    return f"hsl({format_number(hue)} {format_number(saturation)}% {format_number(lightness)}%/{opacity})"


def alpha_forms(rgb, alpha):
    """The rgb() and hsl() texts of a color with alpha, those whose numbers are exact decimals."""
    try:
        opacity = format_number(EXACT.divide(alpha, 255))
    except Inexact:
        return []
    return [form for form in ("rgb(%d %d %d/%s)" % (*rgb, opacity), hsl_text(rgb, opacity)) if form]


def shortest_color(text, round_alpha=False):
    """The shortest exact text of the color; with `round_alpha`, of the color with its alpha rounded to the nearest 1/255."""
    color = parse_color(text.strip())
    if color is None:
        return text
    *rgb, alpha = color
    if round_alpha:
        alpha = alpha.to_integral_value()
    candidates = [(len(text), 9, text)]
    if alpha == alpha.to_integral_value():
        digits = "%02x%02x%02x" % tuple(rgb) + ("" if alpha == 255 else "%02x" % int(alpha))
        candidates.append((len(digits) + 1, 0, "#" + digits))
        if digits[::2] == digits[1::2]:
            candidates.append((len(digits) // 2 + 1, 0, "#" + digits[::2]))
        candidates += [(len(name), 1, name) for name in KEYWORDS_BY_HEX.get("#" + digits, [])]
    if alpha != 255:
        candidates += [(len(form), rank, form) for rank, form in enumerate(alpha_forms(rgb, alpha), 2)]
    return min(candidates)[2]


def is_zero_length(text):
    return ZERO_LENGTH.fullmatch(text) is not None


def shortest_length(text):
    match = LENGTH.fullmatch(text.strip())
    if match is None or re.search(r"\.(?![0-9])", match.group(1)):
        return text
    number, unit = match.groups()
    return format_number(parse_number(number)) + ("" if unit == "px" else unit or "")


def parse_style(text):
    if "\\" in text or "/*" in text or IMPORTANT.search(text):
        return None
    declarations = []
    start = 0
    quote = None
    depth = 0
    for index, char in enumerate(text + ";"):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif char == ";" and depth == 0:
            declaration = text[start:index]
            start = index + 1
            if not declaration.strip(CSS_SPACE):
                continue
            name, colon, value = declaration.partition(":")
            name = name.strip(CSS_SPACE).lower()
            value = value.strip(CSS_SPACE)
            if not colon or not name or not value:
                return None
            declarations.append((name, value))
    if quote or depth:
        return None
    return declarations


def format_style(declarations):
    return ";".join(f"{name}:{value}" for name, value in declarations)


def set_property(element, name, value):
    """Sets a presentation property where the element declares it: in its style, else as an attribute."""
    declarations = parse_style(element.get("style", "")) or []
    if any(key == name for key, _ in declarations):
        element.set("style", format_style([(key, value if key == name else old) for key, old in declarations]))
    else:
        element.set(name, value)


def is_default(name, value, default):
    if name in COLOR_PROPERTIES:
        return shortest_color(value) == shortest_color(default)
    number = parse_number(value)
    default_number = parse_number(default)
    if number is not None and default_number is not None:
        return number == default_number
    return value == default

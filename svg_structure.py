"""Removal of unused definitions and ids, and shortening of the referenced ids."""
import re
from collections import Counter
from itertools import count, product
from string import ascii_letters

from .svg_document import NAME, SVG_NS, WHITESPACE, XLINK_NS, elements, local_name, namespace, remove_node, style_sets, svg_elements
from .svg_transform import parse_transform
from .svg_values import INHERITED_PROPERTIES, parse_style

DEFINITIONS = {"linearGradient", "radialGradient", "pattern", "clipPath", "mask", "marker", "symbol", "filter"}
KEPT = {"title", "desc", "metadata", "font", "font-face", "color-profile"}
HREFS = ("href", f"{{{XLINK_NS}}}href")
ARIA_LISTS = ("aria-labelledby", "aria-describedby")
URL_REFERENCE = re.compile(r"""url\([ \t\r\n\f]*(?:"#([^"]*)"|'#([^']*)'|#([^\s)"']*))[ \t\r\n\f]*\)""", re.IGNORECASE)
NON_GRAPHICS = {"title", "desc", "metadata"}
MERGEABLE_CHILDREN = {"g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "image", "use", "a", "switch"}


def references(element):
    """Every id name the attributes of `element` refer to, once per reference."""
    for key, value in element.attrib.items():
        if key == "id":
            continue
        for match in URL_REFERENCE.finditer(value):
            yield next(group for group in match.groups() if group is not None)
        if key in HREFS and value.startswith("#"):
            yield value[1:]
        elif key in ARIA_LISTS:
            yield from NAME.findall(value)


def short_names(skipped):
    for length in count(1):
        for letters in product(ascii_letters, repeat=length):
            name = "".join(letters)
            if name not in skipped:
                yield name


def is_empty(element):
    return len(element) == 0 and not (element.text or "").strip(WHITESPACE)


def is_removable(element, needed):
    if namespace(element) != SVG_NS or element in needed:
        return False
    name = local_name(element)
    if name in KEPT:
        return False
    parent = element.getparent()
    if name == "g" and parent is not None and local_name(parent) == "switch":
        return False
    if name in DEFINITIONS or (parent is not None and local_name(parent) == "defs" and namespace(parent) == SVG_NS):
        return True
    return name in ("g", "defs") and is_empty(element)


def clean_ids(root):
    if any(local_name(element) == "foreignObject" for element in elements(root)):
        return
    while True:
        counts = Counter(name for element in elements(root) for name in references(element))
        defined = {}
        for element in elements(root):
            if "id" in element.attrib:
                defined.setdefault(element.get("id"), element)
        needed = set()
        for name in counts:
            if name in defined:
                needed.add(defined[name])
                needed.update(defined[name].iterancestors())
        removable = [element for element in elements(root) if element is not root and is_removable(element, needed)]
        if not removable:
            break
        for element in removable:
            remove_node(element)
    order = {element: index for index, element in enumerate(elements(root))}
    used = sorted((name for name in counts if name in defined), key=lambda name: (-counts[name], order[defined[name]]))
    new_names = dict(zip(used, short_names(set(counts) - set(defined))))
    for element in elements(root):
        if "id" in element.attrib:
            name = element.get("id")
            if name in new_names and defined[name] is element:
                element.set("id", new_names[name])
            else:
                del element.attrib["id"]
        for key, value in element.attrib.items():
            if key == "id":
                continue
            rewritten = URL_REFERENCE.sub(lambda match: f"url(#{new_names[name]})" if (name := next(group for group in match.groups() if group is not None)) in new_names else match.group(), value)
            if key in HREFS and value.startswith("#") and value[1:] in new_names:
                rewritten = "#" + new_names[value[1:]]
            elif key in ARIA_LISTS:
                rewritten = NAME.sub(lambda match: new_names.get(match.group(), match.group()), value)
            if rewritten != value:
                element.set(key, rewritten)


def element_children(group):
    return [child for child in group if isinstance(child.tag, str)]


def hoist_common_attributes(group):
    children = [child for child in element_children(group) if local_name(child) not in NON_GRAPHICS]
    if len(children) < 2 or "style" in group.attrib or any("id" in child.attrib for child in children):
        return
    styled = set()
    for child in children:
        declarations = parse_style(child.get("style", ""))
        if declarations is None:
            return
        styled.update(name for name, _ in declarations)
    for name in sorted(INHERITED_PROPERTIES - styled - set(group.attrib)):
        value = children[0].get(name)
        if value is not None and all(child.get(name) == value for child in children):
            group.set(name, value)
            for child in children:
                del child.attrib[name]


def unwrap(group):
    parent = group.getparent()
    children = list(group)
    text = group.text or ""
    if children:
        children[-1].tail = (children[-1].tail or "") + (group.tail or "")
    else:
        text += group.tail or ""
    if text:
        previous = group.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + text
        else:
            parent.text = (parent.text or "") + text
    for child in children:
        group.addprevious(child)
    parent.remove(group)


def can_merge_into(group, child):
    if namespace(child) != SVG_NS or local_name(child) not in MERGEABLE_CHILDREN or "id" in child.attrib:
        return False
    if style_sets(child, group.attrib) or any(name in child.attrib for name in group.attrib if name != "transform"):
        return False
    return "transform" not in group.attrib or "transform" not in child.attrib \
        or (parse_transform(group.get("transform")) is not None and parse_transform(child.get("transform")) is not None)


def collapse_groups(root):
    groups = [element for element in svg_elements(root) if element is not root and local_name(element) == "g"]
    for group in reversed(groups):
        parent = group.getparent()
        if namespace(parent) == SVG_NS and local_name(parent) in ("switch", "clipPath"):
            continue
        hoist_common_attributes(group)
        children = element_children(group)
        if not group.attrib:
            if not any(local_name(child) in NON_GRAPHICS for child in children):
                unwrap(group)
            continue
        if len(children) != 1 or any(name != "transform" and name not in INHERITED_PROPERTIES for name in group.attrib) or not can_merge_into(group, children[0]):
            continue
        child = children[0]
        for name, value in group.attrib.items():
            if name == "transform" and "transform" in child.attrib:
                child.set(name, f"{value} {child.get(name)}")
            else:
                child.set(name, value)
        unwrap(group)

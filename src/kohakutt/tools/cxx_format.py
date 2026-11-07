"""C / C++ formatter: clang-format, then rules it cannot express (overlong values as
``= (`` / value / ``);``, ``;`` in empty loops, ``value,  // name`` arguments).
Usage: python -m kohakutt.tools.cxx_format [--check] [paths...]
"""

import argparse
import functools
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COLUMNS = 88
INDENT = 4
SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}

# A statement assigning (or returning) a value: indent, head (target and operator).
ASSIGN = re.compile(
    r"^(\s*)(return|(?!(?:if|for|while|switch|case)\b)[^=;\"'({]*?[^=!<>\s(]\s*"
    r"(?:<<|>>|[-+*/%&|^])?=)\s+(\S.*)$"
)
# A declaration clang-format split after its type, and the line naming it.
TYPE_ONLY = re.compile(
    r"^\s*(?:(?:const|constexpr|static|unsigned|auto)\s+)*[\w:<>]+[&*]?$"
)
NAMED_VALUE = re.compile(r"^\s*\w+ = \S")
OWN_LINE_ARG = re.compile(r"^(\s*)/\*(\w+)=\*/(.*?)(,?)\s*$")
EMPTY_LOOP = re.compile(r"^(\s*)(?:while|for) \(.*\) \{$")


def clang_format_binary():
    """clang-format beside the running Python (the dev venv), else on PATH."""
    local = Path(sys.executable).with_name("clang-format")
    return str(local) if local.exists() else shutil.which("clang-format")


def scan(line):
    """(code, has_comment): the line with literals and a // comment blanked."""
    out, quote, i = [], None, 0
    while i < len(line):
        c = line[i]
        if quote:
            out.append(" ")
            if c == "\\" and i + 1 < len(line):
                out.append(" ")
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            out.append(" ")
        elif line.startswith("//", i):
            # The comment blanked; a trailing newline kept (positions line up).
            tail = line[i:]
            newline = "\n" if tail.endswith("\n") else ""
            out.append(" " * (len(tail) - len(newline)) + newline)
            return "".join(out), True
        else:
            out.append(c)
        i += 1
    return "".join(out), False


def statement_end(lines, start):
    """Index of the line ending the statement starting at `start` (its ; at bracket
    depth 0); None for anything else (braces anywhere, macros, comments, no end)."""
    depth = 0
    for i in range(start, min(start + 200, len(lines))):
        code, comment = scan(lines[i].rstrip("\n"))
        code = code.rstrip()
        if comment or code.endswith("\\"):
            return None
        for c in code:
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif c in "{}":
                return None
        if depth == 0 and code.endswith(";"):
            return i
    return None


def join(parts):
    """The lines of one expression as one line."""
    out = ""
    for part in parts:
        part = part.strip()
        glued = part.startswith((")", "]", ".", "->", "::")) or out.endswith(
            ("(", "[", "::")
        )
        if out and not glued:
            out += " "
        out += part
    return out


def top_level_comma(text):
    """True when `text` has a comma outside brackets (several declarators)."""
    depth = 0
    for c in scan(text)[0]:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            return True
    return False


def strip_outer_parens(text):
    """`text` without one pair of parentheses enclosing all of it."""
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for i, c in enumerate(scan(text)[0]):
        depth += c == "("
        depth -= c == ")"
        if depth == 0 and i < len(text) - 1:
            return text
    return text[1:-1].strip()


@functools.cache
def config_dir(columns):
    """A directory holding the repository's .clang-format at ColumnLimit `columns`."""
    base = (ROOT / ".clang-format").read_text()
    path = Path(tempfile.mkdtemp(prefix="ktt-cxx-format-"))
    text = re.sub(r"(?m)^ColumnLimit:.*$", f"ColumnLimit: {columns}", base)
    (path / ".clang-format").write_text(text)
    return path


def clang_format(text, binary, columns=COLUMNS):
    """clang-format's output for C++ `text` at `columns`."""
    return subprocess.run(
        [binary, f"--assume-filename={config_dir(columns) / 'x.cpp'}"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def format_value(value, indent, binary):
    """clang-format's layout of expression `value` starting at column `indent`, as the
    sole argument of a call (operands of a broken expression stay at one indent)."""
    snippet = f"void f() {{\n    ktt_value_({value});\n}}\n"
    columns = COLUMNS - indent + 2 * INDENT
    lines = clang_format(snippet, binary, columns).splitlines()[1:-1]
    first, last = lines[0].strip(), lines[-1].strip()
    arg = " " * (2 * INDENT)
    if len(lines) == 1:
        # Everything on one line: ktt_value_(value);
        body = [arg + first[len("ktt_value_(") : -2]]
    elif first == "ktt_value_(":
        # The value on its own lines between ktt_value_( and );.
        body = lines[1:-1]
    else:
        # The value hugged: ktt_value_(call( ... )); its inner lines one indent deeper.
        inner = [" " * INDENT + line for line in lines[1:-1]]
        body = [arg + first[len("ktt_value_(") :], *inner, arg + last[:-2]]
    pad = " " * (indent - 2 * INDENT)
    return [pad + line + "\n" for line in body]


def unwrap_braced_values(lines):
    """Parentheses around an assigned / returned value containing braces (a lambda,
    an initializer) removed: such values keep clang-format's own layout."""
    text = "".join(lines)
    code = "".join(scan(line)[0] for line in text.splitlines(keepends=True))
    out, last = [], 0
    for m in re.finditer(r"(?:\breturn|[^=!<>]=)\s*\(", code):
        open_at = m.end() - 1
        depth, close_at, braced = 0, None, False
        for k in range(open_at, len(code)):
            c = code[k]
            depth += c == "("
            depth -= c == ")"
            braced |= c == "{"
            if depth == 0:
                close_at = k
                break
        after = code[close_at + 1 :].lstrip() if close_at else ""
        if close_at and braced and after.startswith(";") and open_at >= last:
            out += [text[last:open_at], text[open_at + 1 : close_at]]
            last = close_at + 1
    out.append(text[last:])
    return "".join(out).splitlines(keepends=True)


def merge_split_declarations(lines):
    """A type-only line joined with the ``name = value`` line after it."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if TYPE_ONLY.match(line) and NAMED_VALUE.match(nxt):
            out.append(f"{line} {nxt.lstrip()}")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out


def wrap_values(lines, binary):
    """Assigned / returned values that span lines or overflow the line as ``= (`` /
    value / ``);``."""
    lines = merge_split_declarations(unwrap_braced_values(lines))
    out, i = [], 0
    while i < len(lines):
        m = ASSIGN.match(lines[i].rstrip("\n"))
        if m and re.match(r"(using|typedef|template|namespace)\b", m[2]):
            m = None  # a type alias or declaration, not a value
        end = statement_end(lines, i) if m else None
        fits = len(lines[i].rstrip("\n")) <= COLUMNS
        if end is None or (end == i and fits):
            out.append(lines[i])
            i += 1
            continue
        indent, head, first = m.groups()
        rest = (lines[k].rstrip("\n") for k in range(i + 1, end + 1))
        value = strip_outer_parens(join([first, *rest])[:-1].rstrip())
        if value.startswith(("[", "{")) or top_level_comma(value):
            out.extend(lines[i : end + 1])
        elif len(indent) + len(head) + len(value) + 2 <= COLUMNS:
            out.append(f"{indent}{head} {value};\n")
        else:
            inner = len(indent) + INDENT
            if inner + len(value) <= COLUMNS:
                body = [" " * inner + value + "\n"]
            else:
                body = format_value(value, inner, binary)
            out += [f"{indent}{head} (\n", *body, f"{indent});\n"]
        i = end + 1
    return out


def split_items(text):
    """`text` split at commas outside brackets, items stripped."""
    items, depth, start = [], 0, 0
    for i, c in enumerate(scan(text)[0]):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            items.append(text[start:i].strip())
            start = i + 1
    items.append(text[start:].strip())
    return [item for item in items if item]


def hug_braced_lists(lines):
    """A braced list that is a call's only argument opens on the call's line:
    ``f({`` / one element per line / ``});``."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not (
            line.endswith("(") and not line.endswith("= (") and nxt.startswith("{")
        ):
            out.append(lines[i])
            i += 1
            continue

        # The list: from the next line to its matching brace, then the call's close.
        depth, j, text = 0, i + 1, ""
        while j < len(lines):
            code, comment = scan(lines[j].rstrip("\n"))
            if comment:
                break
            depth += code.count("{") - code.count("}")
            text = join([text, lines[j]]) if text else lines[j].strip()
            if depth == 0:
                break
            j += 1
        close = lines[j + 1].strip() if j + 1 < len(lines) else ""
        indent = re.match(r"\s*", line).group(0)
        inner = len(indent) + INDENT
        items = split_items(text[1:-1]) if text.endswith("}") and depth == 0 else []
        fits = all(inner + len(item) + 1 <= COLUMNS for item in items)
        if not items or not fits or not close.startswith(")"):
            out.append(lines[i])
            i += 1
            continue
        out.append(line + "{\n")
        out += [" " * inner + item + ",\n" for item in items[:-1]]
        out.append(" " * inner + items[-1] + "\n")
        out.append(indent + "}" + close + "\n")
        i = j + 2
    return out


def matching_brace(text):
    """Index of the brace closing `text`'s opening brace (text[0] == "{"), or -1."""
    depth = 0
    for i, c in enumerate(scan(text)[0]):
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            return i
    return -1


def block_braced_lists(lines):
    """A braced list clang-format aligns after its brace (``{a,`` / `` b}``) laid out
    as ``{`` / one element per line / ``}`` (then whatever followed the list)."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.lstrip()
        code, comment = scan(line)
        opens = stripped.startswith("{") and stripped != "{"
        if not opens or comment or code.count("{") <= code.count("}"):
            out.append(lines[i])
            i += 1
            continue

        # The list's lines up to its matching brace; `rest` follows it on that line.
        depth, j, text = 0, i, ""
        while j < len(lines):
            code, comment = scan(lines[j].rstrip("\n"))
            if comment:
                break
            depth += code.count("{") - code.count("}")
            text = join([text, lines[j]]) if text else lines[j].strip()
            if depth == 0:
                break
            j += 1
        close = matching_brace(text) if depth == 0 else -1
        indent = line[: len(line) - len(stripped)]
        inner = len(indent) + INDENT
        rest_ok = close > 0 and re.fullmatch(r"[)\]};,\s]*", text[close + 1 :])
        items = split_items(text[1:close]) if rest_ok else []
        if not items or any(inner + len(item) + 1 > COLUMNS for item in items):
            out.append(lines[i])
            i += 1
            continue
        out.append(indent + "{\n")
        out += [" " * inner + item + ",\n" for item in items[:-1]]
        out.append(" " * inner + items[-1] + "\n")
        out.append(indent + text[close:] + "\n")
        i = j + 1
    return out


def house_rules(lines, binary):
    """Argument comments, empty loops, braced lists and value wraps on clang-format's
    output."""
    fixed = []
    for i, line in enumerate(lines):
        m = OWN_LINE_ARG.match(line)
        if m:
            indent, name, value, comma = m.groups()
            line = f"{indent}{value}{comma}  // {name}\n"
        fixed.append(line)
        loop = EMPTY_LOOP.match(line)
        if loop and i + 1 < len(lines) and lines[i + 1].strip() == "}":
            fixed.append(f"{loop.group(1)}{' ' * INDENT};\n")
    return wrap_values(block_braced_lists(hug_braced_lists(fixed)), binary)


def format_text(text, binary, passes=4):
    """The formatted text of one source file: the passes repeated to a fixed point."""
    for _ in range(passes):
        lines = clang_format(text, binary).splitlines(keepends=True)
        formatted = "".join(house_rules(lines, binary))
        if formatted == text:
            break
        text = formatted
    return text


def sources(paths):
    """C / C++ files under `paths`; default: the repository's, outside out/."""
    if paths:
        found = []
        for p in map(Path, paths):
            if p.is_dir():
                found += sorted(f for f in p.rglob("*") if f.suffix in SUFFIXES)
            else:
                found.append(p)
        return found
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.split()
    return [
        ROOT / f
        for f in listed
        if Path(f).suffix in SUFFIXES and not f.startswith("out/")
    ]


def main():
    """Formats files in place; --check lists files that would change (exit 1)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    binary = clang_format_binary()
    changed = []
    for path in sources(args.paths):
        text = path.read_text()
        formatted = format_text(text, binary)
        if formatted != text:
            changed.append(path)
            if not args.check:
                path.write_text(formatted)
    for path in changed:
        print(("would reformat " if args.check else "reformatted ") + str(path))
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())

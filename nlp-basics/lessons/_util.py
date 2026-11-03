"""Console helpers shared by the lessons."""
import textwrap


def title(number: int, name: str) -> None:
    print(f"\n{'=' * 72}\nLESSON {number}: {name}\n{'=' * 72}")


def section(name: str) -> None:
    print(f"\n--- {name} ---")


def say(text: str) -> None:
    print(textwrap.fill(" ".join(text.split()), width=72))


def table(rows, headers) -> None:
    rows = [[f"{c:.3f}" if isinstance(c, float) else str(c) for c in row] for row in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

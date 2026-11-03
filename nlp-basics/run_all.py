"""Run every lesson in order, or a selection:  python run_all.py 1 4 10"""
import importlib
import sys
import time

LESSONS = ["l01_text", "l02_stemming_distance", "l03_bpe", "l04_ngrams", "l05_search",
           "l06_classification", "l07_tagging", "l08_embeddings", "l09_neural", "l10_attention"]


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]} or set(range(1, len(LESSONS) + 1))
    start = time.time()
    for number, name in enumerate(LESSONS, 1):
        if number in wanted:
            importlib.import_module(f"lessons.{name}").main()
    print(f"\nFinished {len(wanted)} lesson(s) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

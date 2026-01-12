import csv
from pathlib import Path

BASE_DIR = Path("data/interim/gdelt_event_context_daily")


def dedupe_file(path: Path) -> None:
    out_path = path.with_name(path.stem + "_deduped.csv")

    seen = set()
    kept = 0
    dropped = 0

    with open(path, "r", newline="", encoding="utf-8") as f_in, open(out_path, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        header = next(reader, None)
        if header is None:
            return

        writer.writerow(header)

        try:
            url_idx = header.index("sourceurl")
        except ValueError:
            print(f"WARNING: no 'sourceurl' column in {path}")
            return

        for row in reader:
            if not row or len(row) <= url_idx:
                continue

            url = row[url_idx].strip()
            if not url:
                continue

            if url in seen:
                dropped += 1
                continue

            seen.add(url)
            writer.writerow(row)
            kept += 1

    print(f"{path.name}: kept {kept:,}, dropped {dropped:,}")


def main():
    files = list(BASE_DIR.rglob("*_event_context.csv"))

    if not files:
        print("No event context CSVs found.")
        return

    for path in sorted(files):
        dedupe_file(path)


if __name__ == "__main__":
    main()

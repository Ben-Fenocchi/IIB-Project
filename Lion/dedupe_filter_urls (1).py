import csv
import re
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path("data/interim/gdelt_event_context_daily")

# Process only one day (YYYYMMDD). Change this to your target date.
ONLY_DAY = "20240303"

# Tune these based on what you see in your URLs
NEGATIVE_PATH_KEYWORDS = [
    # --- SPORTS ---
    "sport","sports","football","soccer","nba","nfl","mlb","nhl","mma","ufc",
    "boxing","wrestling","tennis","golf","cricket","rugby","f1","formula-1",
    "motorsport","nascar","cycling","olympics","athletics","baseball",
    "basketball","hockey","esports","gaming",

    # --- ENTERTAINMENT ---
    "entertainment","celebrity","celebrities","hollywood","bollywood",
    "movies","movie","film","tv","television",
    "streaming","netflix","hulu","prime-video","amazon-prime","disney",
    "disney-plus","hbomax","spotify","music","album","song","songs",
    "concert","tour","festival","theatre","theater","broadway","oscars",
    "emmys","grammys","kardashian","royal-family",

    # --- LIFESTYLE / POP CULTURE ---
    "lifestyle","fashion","beauty","makeup","skincare","hair","diet",
    "fitness","yoga","workout","gym","weightloss","wellness",
    "relationships","dating","wedding","weddings","sex","parenting",
    "horoscope","astrology","zodiac","tarot",

    # --- FOOD / RECIPES ---
    "recipe","recipes","cooking","cook","baking","kitchen","restaurant",
    "food","cuisine","dining","mayo","mayonnaise","chocolate","cake",
    "dessert","wine","beer","cocktail","coffee","tea",

    # --- TRAVEL / TOURISM ---
    "holiday","holidays","vacation","tourism","hotel","hotels",
    "cruise","beach","airport-guide",

    # --- TECH / GADGET REVIEWS ---
    "gadget","gadgets","smartphone","iphone","android",
    "laptop","tablet","camera","headphones","earbuds","tv-review",
    "gaming-console","ps5","xbox","nintendo",

    # --- GENERAL CLICKBAIT ---
    "quiz","giveaway","contest","sweepstakes","lottery",
    "viral","meme","memes","funny","top-10","top10",
    "slideshow","gallery","photos","pictures","wallpaper",

    # --- LOCAL HUMAN INTEREST ---
    "obituary","obituaries","funeral","wedding-announcement","birth",
    "anniversary","missing-person","pet","pets","dog","dogs","cat","cats", "museum"
    , "baby", "babies", "season", "anime", "laundry", "ransom", "scream", "covid", "equaliser",
    "hat-trick","school-shooting", "homicide", "candy", "drugs", "bicycle", "burger", "methamphetamine",
    "cycling", "swimming", "hiking", "pizza", "mcdonalds", "magazine", "turle", "elephant",
    "flower", "balloon", "cinema", "monster", "cult", "meth", "demon", "showbiz", "bishop", "legend", "moon", 
    "queer", "roma", "jewelery", "mom", "dad", "horse", "geforce", "sickle", "casino", "love", "crypto", "seals", "honda", "gender",
    "israel", "gaza", "palestine", "israeli", "genocide", "child", "children", "schools", "epstein", "university", "palestinian", "woman",
    "arts", "fraud", "son", "daughter", "art", "wife", "husband", "gay", "tourism", "rape", "parents", "tvshowbiz", "father", "stabbing", "abortion",
    "suicide", "bitcoin", "birthday", "somalia", "stabbed", "cannabis", "heroin", "biography", "childcare", "motel", "diabetes", "paedophile", "virgin",
    "graduates", "measles", "baptist", "mortgage", "carjacking", "motorcycle", "jewelry", "salmon", "afcon", "robbers", "vaccination", "art", "airbnb", 
    "cooking", "dementia", "spiritual", "holocaust", "pornography", "chelsea", "chatgpt", "childhood", "cryptocurrency", "opioid", "chocolate", "kidnap",
    "wrestling", "pregnancy", "raping", "jews", "raped", "racism", "priests", "bishops", "somali", "porn", "boyfriend", "girlfriend", "somaliland", "tiktok", 
    "fentanyl", "crypto", "marijuana", "antisemitism", "cannabis", "teachers", "podcast", "jewish", "trafficking", "holiday"

]


# Optional: skip obvious non-article pages
NEGATIVE_PATH_PATTERNS = [
    r"/tag/", r"/tags/", r"/category/", r"/author/", r"/gallery/", r"/video/", r"/podcast/"
]

BAD_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".mp4", ".mov", ".avi", ".zip", ".rar", ".7z"
)


def is_irrelevant_url(url: str) -> tuple[bool, str]:
    """
    Conservative URL-only filter.
    Returns (True, reason) if we should drop it.
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return True, "non_http"

    try:
        p = urlparse(url)
        path = (p.path or "").lower()

        # non-article file types
        if path.endswith(BAD_EXTENSIONS):
            return True, "bad_extension"

        # obvious section/category pages
        for pat in NEGATIVE_PATH_PATTERNS:
            if re.search(pat, path):
                return True, f"neg_pattern:{pat}"

        # keyword match in path
        for kw in NEGATIVE_PATH_KEYWORDS:
            if kw in path:
                return True, f"neg_kw:{kw}"

        return False, ""
    except Exception:
        # if parsing fails, keep it (conservative)
        return False, ""


def dedupe_and_filter_file(path: Path) -> None:
    deduped_path = path.with_name(path.stem + "_deduped.csv")
    filtered_path = path.with_name(path.stem + "_deduped_filtered.csv")

    seen = set()
    kept_deduped = 0
    dropped_dupes = 0

    kept_filtered = 0
    dropped_irrelevant = 0

    with open(path, "r", newline="", encoding="utf-8") as f_in, \
         open(deduped_path, "w", newline="", encoding="utf-8") as f_deduped, \
         open(filtered_path, "w", newline="", encoding="utf-8") as f_filtered:

        reader = csv.reader(f_in)
        w_deduped = csv.writer(f_deduped)
        w_filtered = csv.writer(f_filtered)

        header = next(reader, None)
        if header is None:
            return

        w_deduped.writerow(header)
        w_filtered.writerow(header)

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

            # 1) dedupe by URL
            if url in seen:
                dropped_dupes += 1
                continue

            seen.add(url)
            w_deduped.writerow(row)
            kept_deduped += 1

            # 2) filter irrelevant by URL path
            drop, _reason = is_irrelevant_url(url)
            if drop:
                dropped_irrelevant += 1
                continue

            w_filtered.writerow(row)
            kept_filtered += 1

    print(
        f"{path.name}: "
        f"deduped kept {kept_deduped:,}, dupes dropped {dropped_dupes:,} | "
        f"filtered kept {kept_filtered:,}, irrelevant dropped {dropped_irrelevant:,}"
    )
    print(f"  -> {deduped_path.name}")
    print(f"  -> {filtered_path.name}")


def main():
    files = list(BASE_DIR.rglob("*_event_context.csv"))
    if not files:
        print("No event context CSVs found.")
        return

    ran_any = False
    for path in sorted(files):
        if ONLY_DAY and ONLY_DAY not in path.name:
            continue
        ran_any = True
        dedupe_and_filter_file(path)

    if not ran_any:
        print(f"No '*_event_context.csv' files matched ONLY_DAY={ONLY_DAY} under {BASE_DIR}")


if __name__ == "__main__":
    main()

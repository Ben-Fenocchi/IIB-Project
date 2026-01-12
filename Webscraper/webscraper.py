import requests
import trafilatura

# Try to import Newspaper3k as an optional fallback extractor
try:
    from newspaper import Article as _NPArticle
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False


def extract_article_text(url: str, timeout: int = 20) -> dict:
    """
    Extracts and cleans the main article text and title from a URL.
    Returns a dict: {"url": str, "title": str, "text": str}
    Never returns None.
    """

    # Normalise whitespace and remove formatting artefacts
    # so downstream NLP / storage sees clean, comparable text
    def _prep(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\u00a0", " ")  # convert non-breaking spaces to normal spaces
        s = s.replace("\r", " ")
        s = " ".join(s.split())       # collapse runs of whitespace/newlines into single spaces
        return s.strip()

    # Use a realistic browser UA to avoid being blocked or served alternate content
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    # Always initialise outputs so the function never returns None / missing keys
    title, text = "", ""

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        # ---- Primary extractor: Trafilatura ----
        # Optimised for main-body extraction, strips navigation, ads, boilerplate
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        ) or ""

        # Separately extract metadata (title often lives outside main content block)
        meta = trafilatura.bare_extraction(html)
        if isinstance(meta, dict):
            title = meta.get("title") or ""
        else:
            title = getattr(meta, "title", "") or ""

        # ---- Fallback: Newspaper3k if Trafilatura failed or returned empty ----
        if not text and _HAS_NEWSPAPER:
            try:
                art = _NPArticle(url)
                art.download()
                art.parse()
                title = title or (art.title or "")
                text = art.text or ""
            except Exception:
                # If fallback fails, leave title/text as-is and continue
                pass

    except Exception:
        # ---- Final fallback ----
        # Covers network errors, 4xx/5xx, invalid HTML, etc.
        if _HAS_NEWSPAPER:
            try:
                art = _NPArticle(url)
                art.download()
                art.parse()
                title = art.title or ""
                text = art.text or ""
            except Exception:
                pass

    # ---- Clean up title and text before returning ----
    title = _prep(title)
    text = _prep(text)

    return {"url": url, "title": title, "text": text}


if __name__ == "__main__":
    article = extract_article_text("https://www.bbc.co.uk/news/articles/c07m2v1z4evo")

    # Simple sanity check for extraction failure
    if not article["text"]:
        print("Couldn't extract the article body. You can log/skip this URL.")
    else:
        print(article["title"])
        print(article["text"][:500])

import requests
import trafilatura
from datetime import datetime

# Try to import Newspaper3k as an optional fallback extractor
try:
    from newspaper import Article as _NPArticle
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False


def extract_article_text(url: str, timeout: int = 20) -> dict:
    """
    Extracts article text, title, and publish date from a URL.

    Returns a dict with guaranteed keys:
        {
            "url": str,
            "title": str,
            "text": str,
            "publish_date": str | None   # ISO 8601 if available
        }

    publish_date is the ARTICLE publication date, not the event date.
    """

    def _prep(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\u00a0", " ")
        s = s.replace("\r", " ")
        s = " ".join(s.split())
        return s.strip()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    title, text = "", ""
    publish_date = None  # ISO string or None

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        # ---- Primary extractor: Trafilatura ----
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        ) or ""

        meta = trafilatura.bare_extraction(html)

        if isinstance(meta, dict):
            title = meta.get("title") or ""

            # Trafilatura date fields (varies by site)
            raw_date = (
                meta.get("date")
                or meta.get("published")
                or meta.get("datePublished")
            )

            if raw_date:
                try:
                    publish_date = datetime.fromisoformat(
                        raw_date.replace("Z", "+00:00")
                    ).isoformat()
                except Exception:
                    publish_date = None

        # ---- Fallback: Newspaper3k ----
        if (not text or not publish_date) and _HAS_NEWSPAPER:
            try:
                art = _NPArticle(url)
                art.download()
                art.parse()

                title = title or (art.title or "")
                text = text or (art.text or "")

                if art.publish_date:
                    publish_date = art.publish_date.isoformat()

            except Exception:
                pass

    except Exception:
        # ---- Final fallback ----
        if _HAS_NEWSPAPER:
            try:
                art = _NPArticle(url)
                art.download()
                art.parse()

                title = art.title or ""
                text = art.text or ""

                if art.publish_date:
                    publish_date = art.publish_date.isoformat()

            except Exception:
                pass

    title = _prep(title)
    text = _prep(text)

    return {
        "url": url,
        "title": title,
        "text": text,
        "publish_date": publish_date
    }


if __name__ == "__main__":
    article = extract_article_text("https://www.bbc.co.uk/news/articles/c07m2v1z4evo")

    #print("TITLE:", article["title"])
    #print("PUBLISHED:", article["publish_date"])
    #print("TEXT:", article["text"][:500])

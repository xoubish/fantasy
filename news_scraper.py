"""
Sports News & Manager Press Conference Scraper + Risk Analyzer (news_scraper.py)

Fetches recent Premier League / FPL team news from:
- Sporting Life (https://www.sportinglife.com/football/news)
- BBC Sport Premier League (https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml)
- The Guardian Football (https://www.theguardian.com/football/premierleague/rss)
- Sky Sports Football (https://www.skysports.com/football/news)
- Local custom press conference quotes in `my_squad.json` / `data/news_cache.json`

Analyzes press conference quotes for FPL-critical signals:
- Rotation / rest / cameo warnings ("not 100 percent", "needs rest", "failed to start consecutive")
- Injury / suspension flags ("ruled out", "injured", "hamstring", "suspended")
- Positive fitness boosts ("back in training", "100 percent fit", "will start")
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpl_data import DATA_DIR, USER_AGENT, PlayerRecord

NEWS_CACHE_PATH = DATA_DIR / "news_cache.json"

NEWS_SOURCES = [
    {
        "name": "Sporting Life Football",
        "url": "https://www.sportinglife.com/football/news",
        "type": "html",
    },
    {
        "name": "Sporting Life Premier League",
        "url": "https://www.sportinglife.com/football/premier-league",
        "type": "html",
    },
    {
        "name": "BBC Sport Premier League",
        "url": "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml",
        "type": "rss",
    },
    {
        "name": "The Guardian Premier League",
        "url": "https://www.theguardian.com/football/premierleague/rss",
        "type": "rss",
    },
    {
        "name": "Sky Sports Football",
        "url": "https://www.skysports.com/football/news",
        "type": "html",
    },
]

# Keywords that indicate a player is likely benched / rested (high 1-point cameo risk!)
CAMEO_ROTATION_PATTERNS = [
    r"not 100\s*(percent|%)",
    r"needs?\s+a?\s*rest",
    r"\[a rest is\]\s*what he needs",
    r"failed to start consecutive",
    r"manage his minutes",
    r"careful with him",
    r"assess him",
    r"touch and go",
    r"late fitness test",
    r"on the bench",
    r"rotation",
]

# Keywords that indicate a player is completely out (0 minutes)
OUT_INJURED_PATTERNS = [
    r"ruled out",
    r"unavailable",
    r"red[- ]flag",
    r"suspended",
    r"will miss",
    r"out for.*weeks",
    r"surgery",
    r"withdrawn.*warm-up",
]

# Keywords that indicate positive starter confirmation
POSITIVE_STARTER_PATTERNS = [
    r"will start",
    r"fit and ready",
    r"100 percent fit",
    r"back in full training",
    r"key player",
    r"nailed",
]


@dataclass
class NewsArticle:
    source: str
    headline: str
    text: str
    url: str = ""
    player_tag: Optional[str] = None


@dataclass
class PlayerNewsAssessment:
    player_name: str
    risk_level: str  # "NONE", "POSITIVE", "CAMEO_ROTATION_RISK", "INJURED_OR_OUT"
    start_probability: float  # 0.0 to 1.0
    cameo_probability: float  # probability of 1-pt sub appearance (0.0 to 1.0)
    matched_headlines: List[str] = field(default_factory=list)
    matched_quotes: List[str] = field(default_factory=list)
    reasoning: str = "No negative team news."


class _HeadlineHTMLParser(HTMLParser):
    """Lightweight HTML parser to extract article headlines and paragraphs from Sporting Life / Sky."""

    def __init__(self) -> None:
        super().__init__()
        self.in_tag: Optional[str] = None
        self.buffer: List[str] = []
        self.items: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("h1", "h2", "h3", "h4", "p", "a"):
            self.in_tag = tag
            self.buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_tag == tag:
            text = " ".join("".join(self.buffer).split()).strip()
            if len(text) >= 25:
                self.items.append(html.unescape(text))
            self.in_tag = None
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.in_tag:
            self.buffer.append(data)


def _fetch_url(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def scrape_sports_websites() -> Dict[str, Any]:
    """
    Scrapes Sporting Life, BBC Sport, The Guardian, and Sky Sports for recent football news.
    Caches extracted articles to `data/news_cache.json`.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    articles: List[NewsArticle] = []
    errors: List[str] = []

    for src in NEWS_SOURCES:
        try:
            raw = _fetch_url(src["url"])
            if src["type"] == "rss":
                root = ET.fromstring(raw)
                for item in root.findall(".//item")[:25]:
                    title = (item.findtext("title") or "").strip()
                    desc = (item.findtext("description") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    if title:
                        articles.append(
                            NewsArticle(
                                source=src["name"],
                                headline=html.unescape(title),
                                text=html.unescape(re.sub(r"<[^>]+>", "", desc)),
                                url=link,
                            )
                        )
            else:
                parser = _HeadlineHTMLParser()
                parser.feed(raw)
                seen = set()
                for line in parser.items[:40]:
                    if line not in seen:
                        seen.add(line)
                        articles.append(
                            NewsArticle(
                                source=src["name"],
                                headline=line[:140],
                                text=line,
                                url=src["url"],
                            )
                        )
        except Exception as exc:
            errors.append(f"{src['name']}: {exc}")

    if articles:
        NEWS_CACHE_PATH.write_text(
            json.dumps([asdict(a) for a in articles], indent=2), encoding="utf-8"
        )

    return {
        "articles_fetched": len(articles),
        "sources_checked": len(NEWS_SOURCES),
        "errors": errors,
    }


def load_all_news(squad_config_path: Optional[Path] = None) -> List[NewsArticle]:
    """Combines cached web news articles with custom press conference quotes in `my_squad.json`."""
    articles: List[NewsArticle] = []

    if squad_config_path and squad_config_path.exists():
        try:
            cfg = json.loads(squad_config_path.read_text(encoding="utf-8"))
            for item in cfg.get("custom_news", []):
                articles.append(
                    NewsArticle(
                        source=item.get("source", "Manager Press Conference"),
                        headline=item.get("headline", ""),
                        text=item.get("text", ""),
                        player_tag=item.get("player"),
                    )
                )
        except Exception:
            pass

    if NEWS_CACHE_PATH.exists():
        try:
            cached = json.loads(NEWS_CACHE_PATH.read_text(encoding="utf-8"))
            for item in cached:
                articles.append(NewsArticle(**item))
        except Exception:
            pass

    return articles


def _normalize(text: str) -> str:
    return (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("ø", "o")
        .replace("ß", "ss")
        .replace("ã", "a")
        .replace("é", "e")
    )


def analyze_player_news(
    player: PlayerRecord, articles: List[NewsArticle]
) -> PlayerNewsAssessment:
    """
    Evaluates official FPL news + scraped Sporting Life / BBC / press conference quotes
    for a specific player and returns starting vs cameo probabilities.
    """
    p_web = _normalize(player.web_name)
    p_last = _normalize(player.full_name.split()[-1])

    matched_headlines: List[str] = []
    matched_quotes: List[str] = []
    combined_texts: List[str] = []

    # 1. Check Official FPL status & news string
    if player.fpl_news:
        matched_headlines.append(f"[FPL Official] {player.fpl_news}")
        combined_texts.append(player.fpl_news)

    # 2. Match against scraped articles and manager quotes
    for art in articles:
        norm_blob = _normalize(f"{art.headline} {art.text}")
        tagged = art.player_tag and _normalize(art.player_tag) == p_web
        name_in_text = (len(p_web) >= 4 and p_web in norm_blob) or (
            len(p_last) >= 5 and p_last in norm_blob
        )
        if tagged or name_in_text:
            matched_headlines.append(f"[{art.source}] {art.headline}")
            if art.text:
                matched_quotes.append(art.text)
            combined_texts.append(f"{art.headline}. {art.text}")

    full_blob = _normalize(" ".join(combined_texts))

    # 3. Determine Base Start & Cameo Probabilities
    if player.status in ("i", "s", "u") or player.chance_of_playing_next_round == 0:
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="INJURED_OR_OUT",
            start_probability=0.0,
            cameo_probability=0.0,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning="Ruled out (0% chance of playing) due to injury/suspension flag.",
        )

    # Check for rotation / rest / not 100% quotes (like Maresca on O'Reilly)
    cameo_hits = [
        pat for pat in CAMEO_ROTATION_PATTERNS if re.search(pat, full_blob)
    ]
    out_hits = [pat for pat in OUT_INJURED_PATTERNS if re.search(pat, full_blob)]

    if cameo_hits:
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="CAMEO_ROTATION_RISK",
            start_probability=0.20,
            cameo_probability=0.55,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning=(
                "Manager/press quotes indicate player is not 100% or needs rest and is unlikely "
                "to start consecutive matches — high risk of a 1-point bench cameo!"
            ),
        )

    if out_hits and not cameo_hits:
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="INJURED_OR_OUT",
            start_probability=0.05,
            cameo_probability=0.10,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning="News reports suggest player is ruled out or unavailable.",
        )

    if (
        player.chance_of_playing_next_round is not None
        and player.chance_of_playing_next_round < 100
    ):
        prob = (player.chance_of_playing_next_round / 100.0) * 0.7
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="CAMEO_ROTATION_RISK",
            start_probability=round(prob, 2),
            cameo_probability=0.35,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning=f"FPL flags player at {player.chance_of_playing_next_round}% chance of playing.",
        )

    pos_hits = [pat for pat in POSITIVE_STARTER_PATTERNS if re.search(pat, full_blob)]
    if pos_hits:
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="POSITIVE",
            start_probability=0.95,
            cameo_probability=0.03,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning="Positive fitness/starting confirmation in team news.",
        )

    # Default healthy player: scale start probability by actual season minutes/starts
    if player.minutes < 90:
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="BACKUP_LOW_MINS",
            start_probability=0.05,
            cameo_probability=0.10,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning=f"Backup squad player ({player.minutes} mins played this season).",
        )
    if player.minutes < 360:
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="ROTATION_RISK",
            start_probability=0.45,
            cameo_probability=0.30,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning=f"Rotation player ({player.minutes} mins played across {player.starts} starts).",
        )

    base_start = 0.94 if player.minutes >= 630 else 0.80
    return PlayerNewsAssessment(
        player_name=player.web_name,
        risk_level="NONE",
        start_probability=base_start,
        cameo_probability=round((1.0 - base_start) * 0.5, 2),
        matched_headlines=matched_headlines,
        matched_quotes=matched_quotes,
        reasoning="Regular starter with no negative press conference or injury flags.",
    )

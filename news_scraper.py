"""
Sports News & Manager Press Conference Scraper + Risk Analyzer (news_scraper.py)

Fetches recent Premier League / FPL team news from:
- Sporting Life (https://www.sportinglife.com/football/news)
- BBC Sport Premier League (https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml)
- The Guardian Football (https://www.theguardian.com/football/premierleague/rss)
- Sky Sports Football (https://www.skysports.com/football/news)
- Local custom press conference quotes in `my_squad.json` / `data/news_cache.json`

Availability (minutes) is estimated by the learned minutes model in `neural_model.py`
plus the OFFICIAL FPL flags. Keyword phrases below are applied ONLY to notes you add
yourself in `my_squad.json` (tagged to a player); scraped headlines are shown for
information and never move the probabilities.
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fpl_data import DATA_DIR, USER_AGENT, PlayerRecord

NEWS_CACHE_PATH = DATA_DIR / "news_cache.json"

# Backtests inject a model trained only on past gameweeks here so availability never
# uses weights that have seen the week being tested.
MINUTES_MODEL_OVERRIDE = None

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
    return_date: Optional[str] = None  # ISO date parsed from the official "Expected back"/"until" text


_RETURN_DATE = re.compile(r"(?:expected back|back on|until|return(?:s|ing)?(?: on)?)\s+(\d{1,2})\s+([A-Za-z]{3})[a-z]*", re.I)
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}


def parse_return_date(news: str, reference: Optional[datetime] = None) -> Optional[datetime]:
    """Date in the official FPL news text ("Hamstring injury - Expected back 22 Nov", "Suspended until 29 Nov").
    The year is the one that puts the date within four months before, or any time after, `reference`."""
    if not news:
        return None
    match = _RETURN_DATE.search(news)
    if not match or match.group(2).lower() not in _MONTHS:
        return None
    reference = reference or datetime.now(timezone.utc)
    day, month = int(match.group(1)), _MONTHS[match.group(2).lower()]
    try:
        candidate = datetime(reference.year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None
    if candidate < reference - timedelta(days=120):
        candidate = candidate.replace(year=reference.year + 1)
    return candidate


def kickoff_datetime(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


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
    player: PlayerRecord,
    articles: List[NewsArticle],
    minutes_probs: Optional[Tuple[float, float]] = None,
) -> PlayerNewsAssessment:
    """
    Availability assessment = learned minutes model + OFFICIAL FPL flags + user-curated notes.

      * `minutes_probs` = (P(60+ minutes), P(1-59 minutes)) from the neural minutes head.
        When not supplied it is computed here from the cached model.
      * Official FPL status / chance_of_playing_next_round overrides the model.
      * Custom notes in `my_squad.json` that are explicitly tagged with this player's name
        are honoured (rest / out / confirmed-start phrases).
      * Scraped headlines are matched by name for DISPLAY ONLY. They never change the
        probabilities: loose keyword matching on headlines proved to be noise.
    """
    p_web = _normalize(player.web_name)

    matched_headlines: List[str] = []
    matched_quotes: List[str] = []
    curated_texts: List[str] = []

    if player.fpl_news:
        matched_headlines.append(f"[FPL Official] {player.fpl_news}")

    for art in articles:
        tagged = bool(art.player_tag) and _normalize(art.player_tag) == p_web
        if tagged:
            matched_headlines.append(f"[{art.source}] {art.headline}")
            if art.text:
                matched_quotes.append(art.text)
            curated_texts.append(f"{art.headline}. {art.text}")
            continue
        # informational name match on the headline only (full web_name, >= 5 chars)
        if len(p_web) >= 5 and p_web in _normalize(art.headline):
            matched_headlines.append(f"[{art.source}] {art.headline}")

    # 1. Hard official flags, read together with the official return date when one is given
    return_dt = parse_return_date(player.fpl_news)
    return_iso = return_dt.date().isoformat() if return_dt else None
    next_fixture = next((f for f in player.upcoming_3gw_fixtures if int(f.get("gw", 0) or 0) == int(player.target_gw or 0)), None)
    next_kickoff = kickoff_datetime(next_fixture.get("kickoff")) if next_fixture else None
    back_before_kickoff = bool(return_dt and next_kickoff and return_dt.date() <= next_kickoff.date())
    flagged_out = player.status in ("i", "s", "u") or player.chance_of_playing_next_round == 0
    if flagged_out and not (back_before_kickoff and player.status != "u"):
        return PlayerNewsAssessment(
            player_name=player.web_name,
            risk_level="INJURED_OR_OUT",
            start_probability=0.0,
            cameo_probability=0.0,
            matched_headlines=matched_headlines,
            matched_quotes=matched_quotes,
            reasoning="Ruled out by the official FPL injury/suspension flag."
                      + (f" Expected back {return_iso}." if return_iso else ""),
            return_date=return_iso,
        )

    # 2. Learned minutes model
    if minutes_probs is None:
        try:
            from neural_model import extract_features, get_or_train_neural_net

            net = MINUTES_MODEL_OVERRIDE or get_or_train_neural_net()
            full = net.predict_full(extract_features(player))
            minutes_probs = (full["p_full"], full["p_cameo"])
        except Exception:
            rate = min(1.0, player.minutes / max(90.0 * max(1, len(player.gw_history)), 90.0))
            minutes_probs = (0.9 * rate, 0.1)
    p_full, p_cameo = float(minutes_probs[0]), float(minutes_probs[1])
    p_full = max(0.0, min(1.0, p_full))
    p_cameo = max(0.0, min(1.0 - p_full, p_cameo))
    reason = f"Minutes model: {int(p_full * 100)}% to play 60+, {int(p_cameo * 100)}% cameo."

    # 3. Official doubt flags scale the model
    chance = player.chance_of_playing_next_round
    if flagged_out and back_before_kickoff:
        # Still flagged, but the official text expects a return before this kickoff: first game back.
        p_cameo = min(1.0 - p_full * 0.5, p_cameo + p_full * 0.25)
        p_full *= 0.5
        reason = f"Official text expects a return on {return_iso}, before kickoff; treated as a first game back; " + reason
    elif chance is not None and chance < 100:
        c = chance / 100.0
        p_cameo = min(1.0 - p_full * c, p_cameo * c + p_full * (1.0 - c) * 0.35)
        p_full = p_full * c
        reason = f"FPL flags {chance}% chance of playing; " + reason
    elif player.status == "d":
        p_cameo = min(1.0 - p_full * 0.5, p_cameo + p_full * 0.2)
        p_full *= 0.5
        reason = "FPL marks player doubtful; " + reason

    # 4. User-curated notes explicitly tagged to this player
    blob = _normalize(" ".join(curated_texts))
    if blob:
        if any(re.search(pat, blob) for pat in OUT_INJURED_PATTERNS):
            p_full, p_cameo = min(p_full, 0.05), min(1.0 - 0.05, max(p_cameo, 0.10))
            reason = "Your note says the player is out; " + reason
        elif any(re.search(pat, blob) for pat in CAMEO_ROTATION_PATTERNS):
            p_full, p_cameo = min(p_full, 0.25), max(p_cameo, 0.45)
            reason = "Your note flags rest/rotation (cameo trap); " + reason
        elif any(re.search(pat, blob) for pat in POSITIVE_STARTER_PATTERNS):
            p_full = max(p_full, 0.90)
            p_cameo = min(p_cameo, 1.0 - p_full)
            reason = "Your note confirms a start; " + reason

    p_play = p_full + p_cameo
    if p_play < 0.15:
        risk = "BACKUP_LOW_MINS"
    elif p_full >= 0.72:
        risk = "NONE"
    elif p_full >= 0.45:
        risk = "ROTATION_RISK"
    else:
        risk = "CAMEO_ROTATION_RISK"

    return PlayerNewsAssessment(
        player_name=player.web_name,
        risk_level=risk,
        start_probability=round(p_full, 3),
        cameo_probability=round(p_cameo, 3),
        matched_headlines=matched_headlines,
        matched_quotes=matched_quotes,
        reasoning=reason,
        return_date=return_iso,
    )

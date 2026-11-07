# FPL AI Squad Optimizer & Variation Explorer

A lightweight Python + Web toolkit that combines **3-year historical FPL data (`2023-24` to `2025-26`)**, **current season underlying metrics (`xG/90`, `xA/90`, `xGC/90`, `BPS/90`)**, **Poisson clean-sheet & goals-conceded models**, and **live sports news / manager press conference scraping** (Sporting Life, BBC Sport, Guardian, Sky Sports) to build optimal Fantasy Premier League squads before each Gameweek deadline.

## Features

- **Official FPL Scoring & Squad Rules Engine (`predictor.py`)**:
  - Position-weighted goal points (`GKP: 10`, `DEF: 6`, `MID: 5`, `FWD: 4`), clean sheets (`GKP/DEF: +4`, `MID: +1`), multi-tier goals conceded penalties (`-1 pt` per 2 goals conceded), dynamic save points (`+1 pt` per 3 saves), card/penalty-miss adjustments, and Captain (`2x`) + Vice-Captain inheritance.
- **Press Conference & 1-Point Cameo Trap Detection (`news_scraper.py`)**:
  - Flags players whose managers mention *"not 100 percent"*, *"needs rest"*, or *"failed to start consecutive matches"* so a 10-minute bench cameo doesn't block your auto-substitutions.
- **Unbiased Multi-Variation £100m Squad Generator (`get_best_squad_variations`)**:
  - Generates **6 distinct tactical £100.0m 15-player squads** (`Balanced 3-5-2`, `Haaland+Saka+Palmer Threemium 3-4-3`, `Heavy Metal Front Three 3-4-3`, `Clean-Sheet Fortress 4-4-2`, `Differential Wingers 3-5-2`, and `Attacking Fullbacks + Haaland 4-3-3`) plus an on-the-fly **Random 70–80p Squad Shuffler**.
- **Minimal Interactive Web Page (`index.html` & `web_app.py`)**:
  - Interactive pitch visualization with club kits, Captain/Vice badges, Player Inspector panel, and buttons to cycle through tactical variations or shuffle new 70–80p squads.

## Quick Start

```bash
# 1. Sync live FPL API data, 3-year historical stats, and sports news
python3 fpl_assistant.py sync

# 2. Analyze your squad in my_squad.json and generate PNG pitch figures
python3 fpl_assistant.py analyze

# 3. Generate the interactive web page (index.html)
python3 web_app.py

# 4. Or serve it locally at http://localhost:8000
python3 web_app.py --serve
```

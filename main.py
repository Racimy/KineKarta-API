import json
import re
import time
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
from html.parser import HTMLParser
from urllib.parse import urljoin
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from auth import (
    UserRegister, UserLogin, Token, UserPreferences, UserProfile, LetterboxdData,
    create_user, authenticate_user, create_access_token,
    get_current_user, update_user_preferences, update_user_letterboxd, get_user_letterboxd
)
import math
import httpx
import uvicorn
import os
import asyncio
from zip_coords import LOCATION_COORDS

app = FastAPI(title="Kinekarta API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = "https://paris-cine.info"
TMDB_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1MzdiNGZkN2MxMTI4YmY0YTc4YWJhNTI1ZGU4MzJjNiIsIm5iZiI6MTc2NTI2Nzc3Ni42MjEsInN1YiI6IjY5MzdkOTQwMTVmMDc3NWQ5ZGViMWJmYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.QJ-fT1iFjpaHEBXTP3Q4CaxCpN2fSPUOC_aeH3e-Y1I"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "537b4fd7c1128bf4a78aba525de832c6")
TMDB_LIST_ID = os.getenv("TMDB_LIST_ID", "")
TMDB_V3_BASE = "https://api.themoviedb.org/3"
TMDB_V4_BASE = "https://api.themoviedb.org/4"
FILTERS_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "filters.json")
FILTERS_CACHE_TTL = int(os.getenv("PCI_FILTERS_TTL", "1800"))
FILTERS_CACHE = {"expires": 0.0, "data": None}
FILTERS_LOCK = asyncio.Lock()
FILTER_KEYS = ("cinemas", "locations", "formats", "languages", "cards", "dates", "genres")
FRENCH_DAYS = ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
try:
    FILTERS_TIMEZONE = ZoneInfo(os.getenv("PCI_TIMEZONE", "Europe/Paris"))
except Exception:
    FILTERS_TIMEZONE = None
SHOWTIMES_CACHE_TTL = int(os.getenv("PCI_SHOWTIMES_TTL", "600"))
SHOWTIMES_CACHE = {}
SHOWTIMES_SEMAPHORE = asyncio.Semaphore(int(os.getenv("PCI_SHOWTIMES_MAX_CONCURRENCY", "8")))
TMDB_MAX_CONCURRENCY = int(os.getenv("TMDB_MAX_CONCURRENCY", "6"))
TMDB_SEMAPHORE = asyncio.Semaphore(TMDB_MAX_CONCURRENCY)

FILTERS_REFRESH_TASK: Optional[asyncio.Task] = None
FILTERS_REFRESH_STOP = asyncio.Event()

def normalize_location_key(value: str) -> str:
    if not value:
        return ""
    cleaned = unicodedata.normalize("NFKD", str(value))
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", cleaned).strip().lower()
    return " ".join(cleaned.split())

LOCATION_COORDS_NORMALIZED = {
    normalize_location_key(key): (key, coords)
    for key, coords in LOCATION_COORDS.items()
}

def resolve_location_query(query: str) -> Optional[dict]:
    raw = (query or "").strip()
    if not raw:
        return None

    match = re.search(r"\b(\d{5})\b", raw)
    if match:
        code = match.group(1)
        coords = LOCATION_COORDS.get(code)
        if coords:
            return {"label": code, "lat": coords[0], "lon": coords[1]}
        if code == "75000":
            coords = LOCATION_COORDS.get("Paris Centre") or LOCATION_COORDS.get("Paris 01")
            if coords:
                return {"label": "Paris Centre", "lat": coords[0], "lon": coords[1]}
        if code.startswith("75"):
            try:
                arrondissement = int(code[2:])
            except ValueError:
                arrondissement = 0
            if 1 <= arrondissement <= 20:
                key = f"Paris {arrondissement:02d}"
                coords = LOCATION_COORDS.get(key) or LOCATION_COORDS.get(f"Paris {arrondissement}")
                if coords:
                    return {"label": key, "lat": coords[0], "lon": coords[1]}

    lower = raw.lower()
    if "paris" in lower:
        match = re.search(r"\bparis\s*(\d{1,2})", lower)
        if match:
            arrondissement = int(match.group(1))
            key = f"Paris {arrondissement:02d}"
            coords = LOCATION_COORDS.get(key) or LOCATION_COORDS.get(f"Paris {arrondissement}")
            if coords:
                return {"label": key, "lat": coords[0], "lon": coords[1]}

    normalized = normalize_location_key(raw)
    if normalized == "paris":
        coords = LOCATION_COORDS.get("Paris Centre") or LOCATION_COORDS.get("Paris 01")
        if coords:
            return {"label": "Paris Centre", "lat": coords[0], "lon": coords[1]}
    direct = LOCATION_COORDS_NORMALIZED.get(normalized)
    if direct:
        label, coords = direct
        return {"label": label, "lat": coords[0], "lon": coords[1]}

    best = None
    for key, (label, coords) in LOCATION_COORDS_NORMALIZED.items():
        if key and key in normalized:
            if not best or len(key) > best[0]:
                best = (len(key), label, coords)
    if best:
        _, label, coords = best
        return {"label": label, "lat": coords[0], "lon": coords[1]}

    return None

def cinema_coords_from_group(group: str) -> Optional[tuple[float, float]]:
    raw = (group or "").strip()
    if not raw:
        return None
    coords = LOCATION_COORDS.get(raw)
    if coords:
        return coords
    normalized = normalize_location_key(raw)
    entry = LOCATION_COORDS_NORMALIZED.get(normalized)
    if entry:
        return entry[1]
    best = None
    for key, (_, coords) in LOCATION_COORDS_NORMALIZED.items():
        if key and key in normalized:
            if not best or len(key) > best[0]:
                best = (len(key), coords)
    if best:
        return best[1]
    return None

def cinema_coords_from_cinema(cinema: dict) -> Optional[tuple[float, float]]:
    # Try group first
    group = cinema.get("group")
    coords = cinema_coords_from_group(group)
    if coords:
        return coords
    
    # Try label (city name often in label)
    label = cinema.get("label", "")
    coords = cinema_coords_from_group(label)
    if coords:
        return coords
        
    return None

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Format codes used by the showtimes endpoint.
FORMAT_VALUE_TO_CODE = {
    "dci": "DCI",
    "dat": "DAT",
    "ice": "ICE",
    "scx": "SCX",
    "onx": "ONX",
    "4dx": "4DX",
    "4de": "4DE",
    "excellence": "EXCELLENCE",
}
FORMAT_CODE_TO_VALUE = {v: k for k, v in FORMAT_VALUE_TO_CODE.items()}

# TMDB Cache
CACHE_FILE = os.path.join(os.path.dirname(__file__), "tmdb_cache.json")
tmdb_cache = {}

def load_cache():
    global tmdb_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                tmdb_cache = json.load(f)
        except Exception:
            tmdb_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(tmdb_cache, f, ensure_ascii=False)
    except Exception:
        pass

load_cache()

def empty_filters() -> dict:
    return {key: [] for key in FILTER_KEYS}

def load_filters_fallback() -> dict:
    try:
        with open(FILTERS_FALLBACK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {key: data.get(key, []) for key in FILTER_KEYS}
    except Exception:
        return empty_filters()

def filters_now() -> datetime:
    if FILTERS_TIMEZONE:
        return datetime.now(FILTERS_TIMEZONE)
    return datetime.now()

class SelectOptionParser(HTMLParser):
    def __init__(self, select_id: str):
        super().__init__(convert_charrefs=True)
        self.select_id = select_id
        self.in_select = False
        self.in_option = False
        self.current_value = ""
        self.current_label = []
        self.current_group = None
        self.options = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            sel_id = attrs.get("id") or attrs.get("name")
            self.in_select = sel_id == self.select_id
            if self.in_select:
                self.current_group = None
        elif tag == "optgroup" and self.in_select:
            label = (attrs.get("label") or "").strip()
            self.current_group = label or None
        elif tag == "option" and self.in_select:
            self.in_option = True
            self.current_value = (attrs.get("value") or "").strip()
            self.current_label = []

    def handle_endtag(self, tag):
        if tag == "option" and self.in_option:
            label = "".join(self.current_label).strip()
            value = (self.current_value or "").strip()
            if value and value.lower() != "all":
                self.options.append({
                    "value": value,
                    "label": label or value,
                    "group": self.current_group,
                })
            self.in_option = False
            self.current_value = ""
            self.current_label = []
        elif tag == "optgroup" and self.in_select:
            self.current_group = None
        elif tag == "select" and self.in_select:
            self.in_select = False
            self.current_group = None

    def handle_data(self, data):
        if self.in_option and data:
            self.current_label.append(data)

def dedupe_options(options: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in options:
        value = str(item.get("value", "")).strip()
        if not value or value in seen:
            continue
        label = str(item.get("label") or value).strip()
        group = item.get("group")
        if isinstance(group, str):
            group = group.strip() or None
        deduped.append({"value": value, "label": label, "group": group})
        seen.add(value)
    return deduped

def parse_select_options(html_text: str, select_id: str) -> list[dict]:
    parser = SelectOptionParser(select_id)
    parser.feed(html_text or "")
    return dedupe_options(parser.options)

def parse_filters_from_html(html_text: str) -> dict:
    if not html_text:
        return empty_filters()
    filters = {
        "cinemas": parse_select_options(html_text, "selcine"),
        "locations": parse_select_options(html_text, "seladdr"),
        "cards": parse_select_options(html_text, "selcard"),
        "languages": parse_select_options(html_text, "sellang"),
        "formats": parse_select_options(html_text, "selformat"),
        "dates": parse_select_options(html_text, "selday"),
        "genres": parse_select_options(html_text, "selgenre"),
    }
    return {key: dedupe_options(filters.get(key, [])) for key in FILTER_KEYS}

def merge_option_lists(primary: list[dict], fallback: list[dict]) -> list[dict]:
    merged = []
    index = {}
    for item in fallback:
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        index[value] = {
            "value": value,
            "label": item.get("label") or value,
            "group": item.get("group"),
        }
    for item in primary:
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        base = index.get(value, {})
        merged.append({
            "value": value,
            "label": item.get("label") or base.get("label") or value,
            "group": item.get("group") or base.get("group"),
        })
        index.pop(value, None)
    for item in fallback:
        value = str(item.get("value", "")).strip()
        if value and value in index:
            merged.append(index[value])
            index.pop(value, None)
    return dedupe_options(merged)

def merge_filters(primary: dict, fallback: dict) -> dict:
    merged = {}
    for key in FILTER_KEYS:
        merged[key] = merge_option_lists(primary.get(key, []), fallback.get(key, []))
    return merged

def get_seldayid(selday: str) -> int:
    """Calcule le seldayid (offset depuis aujourd'hui) pour paris-cine.info.

    Paris-cine.info utilise seldayid comme offset:
    - 0 = tous (week)
    - 1 = aujourd'hui
    - 2 = demain
    - 3 = après-demain
    - etc.
    """
    value = (selday or "").strip().lower()
    if not value or value == "week" or value == "all":
        return 0
    if value == "today":
        return 1

    # Pour les jours de la semaine, calculer l'offset
    now = filters_now()
    js_today = (now.weekday() + 1) % 7  # 0=dimanche, 1=lundi, etc.

    day_to_index = {
        "dimanche": 0, "lundi": 1, "mardi": 2, "mercredi": 3,
        "jeudi": 4, "vendredi": 5, "samedi": 6
    }

    target_index = day_to_index.get(value)
    if target_index is None:
        return 0

    # Calculer l'offset (combien de jours jusqu'à ce jour)
    # Si on est mardi (2) et on veut jeudi (4), offset = 4-2 = 2, mais +1 car 1=aujourd'hui
    # Donc jeudi = offset 3
    offset = (target_index - js_today) % 7
    if offset == 0:
        offset = 7  # Le même jour la semaine prochaine
    return offset + 1  # +1 car 1=aujourd'hui


def build_dynamic_date_filters(base_dates: list[dict]) -> list[dict]:
    options = dedupe_options(base_dates)
    by_value = {opt.get("value"): opt for opt in options if opt.get("value")}

    def take(value: str, fallback_label: str) -> dict:
        item = by_value.pop(value, None)
        label = (item.get("label") if item else "") or fallback_label
        return {"value": value, "label": label, "group": None}

    ordered = [
        take("week", "Cette semaine"),
        take("today", "Aujourd'hui"),
    ]

    now = filters_now()
    js_index = (now.weekday() + 1) % 7
    for offset in range(1, 8):
        day_name = FRENCH_DAYS[(js_index + offset) % 7]
        value = day_name.lower()
        if value in by_value:
            item = by_value.pop(value)
            label = item.get("label") or day_name
        else:
            label = day_name
        ordered.append({"value": value, "label": label, "group": None})

    all_label = "Toutes dates"
    if "all" in by_value:
        existing = by_value.pop("all")
        all_label = existing.get("label") or all_label
    ordered.append({"value": "all", "label": all_label, "group": None})

    for item in by_value.values():
        ordered.append(item)

    return dedupe_options(ordered)

def build_showtime_cache_key(mov_id: str, params: dict) -> tuple:
    return (
        str(mov_id),
        str(params.get("selday", "")),
        str(params.get("selcard", "")),
        str(params.get("seladdr", "")),
        str(params.get("selcine", "")),
        str(params.get("sellang", "")),
    )

def get_cached_showtimes(cache_key: tuple) -> list | None:
    entry = SHOWTIMES_CACHE.get(cache_key)
    if not entry:
        return None
    if entry.get("expires", 0) > time.time():
        return entry.get("data", [])
    SHOWTIMES_CACHE.pop(cache_key, None)
    return None

def set_cached_showtimes(cache_key: tuple, showtimes: list) -> None:
    SHOWTIMES_CACHE[cache_key] = {
        "expires": time.time() + max(SHOWTIMES_CACHE_TTL, 0),
        "data": showtimes,
    }

def normalize_request_params(params: dict) -> dict:
    """Normalise les paramètres pour l'API paris-cine.info."""
    result = {}
    selcine = params.get("selcine", "")
    for k, v in params.items():
        val = str(v).strip()
        if k == "selcine":
            result[k] = "" if val.lower() == "all" or not val else val
        elif k == "seladdr":
            if val == "*" or val.lower() == "all" or not val:
                # Si un cinéma est spécifié, on laisse vide (recherche globale)
                # Si aucun cinéma, on met 75000 par défaut car le site distant l'exige souvent pour PCImovies
                if selcine:
                    result[k] = ""
                else:
                    result[k] = "75000"
            else:
                result[k] = val
        elif val.lower() == "all" or not val:
            result[k] = "all"
        else:
            result[k] = val
    return result

def extract_script_urls(html_text: str) -> list[str]:
    urls = []
    for src in re.findall(r"<script[^>]+src=[\"']([^\"']+)[\"']", html_text or "", flags=re.IGNORECASE):
        urls.append(urljoin(f"{BASE_URL}/", src))
    seen = set()
    unique = []
    for url in urls:
        if url in seen:
            continue
        unique.append(url)
        seen.add(url)
    return unique

def extract_js_object(text: str, name: str) -> str:
    if not text:
        return ""
    match = re.search(rf"{re.escape(name)}\s*=\s*{{", text)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return ""

def decode_js_string(value: str) -> str:
    if not value:
        return value
    if "\\u" in value or "\\x" in value:
        try:
            return bytes(value, "utf-8").decode("unicode_escape")
        except Exception:
            return value
    return value

def parse_js_mapping(obj_text: str) -> dict:
    if not obj_text:
        return {}
    mapping = {}
    for match in re.finditer(r"[\"']([^\"']+)[\"']\s*:\s*[\"']([^\"']*)[\"']", obj_text):
        key = match.group(1).strip()
        value = decode_js_string(match.group(2).strip())
        if key:
            mapping[key] = value
    for match in re.finditer(r"([A-Za-z0-9_]+)\s*:\s*[\"']([^\"']*)[\"']", obj_text):
        key = match.group(1).strip()
        if key in mapping:
            continue
        value = decode_js_string(match.group(2).strip())
        if key:
            mapping[key] = value
    if mapping:
        return mapping
    for match in re.finditer(r"[\"']([^\"']+)[\"']\s*:\s*\[([^\]]+)\]", obj_text):
        key = match.group(1).strip()
        raw_values = match.group(2)
        first = re.search(r"[\"']([^\"']+)[\"']", raw_values)
        if key and first:
            mapping[key] = decode_js_string(first.group(1).strip())
    if mapping:
        return mapping
    for match in re.finditer(r"([A-Za-z0-9_]+)\s*:\s*\[([^\]]+)\]", obj_text):
        key = match.group(1).strip()
        if key in mapping:
            continue
        raw_values = match.group(2)
        first = re.search(r"[\"']([^\"']+)[\"']", raw_values)
        if key and first:
            mapping[key] = decode_js_string(first.group(1).strip())
    return mapping

def apply_js_mappings(filters: dict, js_texts: list[str]) -> dict:
    format_labels = {}
    genre_labels = {}
    for text in js_texts:
        format_labels.update(parse_js_mapping(extract_js_object(text, "format_disp")))
        format_labels.update(parse_js_mapping(extract_js_object(text, "format_strings")))
        genre_labels.update(parse_js_mapping(extract_js_object(text, "genre_strings")))
    if format_labels:
        if not filters.get("formats"):
            filters["formats"] = [
                {"value": key, "label": label or key, "group": None}
                for key, label in format_labels.items()
            ]
        else:
            for item in filters["formats"]:
                label = format_labels.get(item.get("value"))
                if label:
                    item["label"] = label
    if genre_labels:
        if not filters.get("genres"):
            filters["genres"] = [
                {"value": key, "label": label or key, "group": None}
                for key, label in genre_labels.items()
            ]
        else:
            for item in filters["genres"]:
                label = genre_labels.get(item.get("value"))
                if label:
                    item["label"] = label
    for key in FILTER_KEYS:
        filters[key] = dedupe_options(filters.get(key, []))
    return filters

async def fetch_filters_from_paris_cine() -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(BASE_URL)
            if resp.status_code != 200:
                return None
            html_text = resp.text
            filters = parse_filters_from_html(html_text)
            script_urls = extract_script_urls(html_text)
            if script_urls:
                preferred = [u for u in script_urls if "main" in u or "async" in u]
                target = preferred or script_urls
                target = target[:3]
                responses = await asyncio.gather(
                    *[client.get(url) for url in target],
                    return_exceptions=True,
                )
                js_texts = []
                for res in responses:
                    if isinstance(res, Exception):
                        continue
                    if res.status_code == 200:
                        js_texts.append(res.text)
                if js_texts:
                    filters = apply_js_mappings(filters, js_texts)
            filters["dates"] = build_dynamic_date_filters(filters.get("dates", []))
            return filters
    except Exception:
        return None

async def refresh_filters(force: bool = False) -> dict:
    now = time.time()
    cached = FILTERS_CACHE.get("data")
    if not force and cached and FILTERS_CACHE.get("expires", 0) > now:
        return cached

    async with FILTERS_LOCK:
        cached = FILTERS_CACHE.get("data")
        if not force and cached and FILTERS_CACHE.get("expires", 0) > now:
            return cached

        remote = await fetch_filters_from_paris_cine()
        fallback = load_filters_fallback()
        if remote:
            merged = merge_filters(remote, fallback)
        else:
            merged = cached or fallback
            
        # Enrichir les cinémas avec leurs coordonnées
        if "cinemas" in merged:
            for cinema in merged["cinemas"]:
                coords = cinema_coords_from_cinema(cinema)
                if coords:
                    cinema["lat"] = coords[0]
                    cinema["lon"] = coords[1]
                    
        merged["dates"] = build_dynamic_date_filters(merged.get("dates", []))
        FILTERS_CACHE["data"] = merged
        FILTERS_CACHE["expires"] = time.time() + max(FILTERS_CACHE_TTL, 0)
        return merged

async def filters_refresh_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        now = filters_now()
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait_seconds = max((next_midnight - now).total_seconds(), 1)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        try:
            await refresh_filters(force=True)
        except Exception:
            continue

@app.on_event("startup")
async def on_startup() -> None:
    global FILTERS_REFRESH_TASK
    await refresh_filters(force=True)
    FILTERS_REFRESH_STOP.clear()
    FILTERS_REFRESH_TASK = asyncio.create_task(filters_refresh_loop(FILTERS_REFRESH_STOP))

@app.on_event("shutdown")
async def on_shutdown() -> None:
    FILTERS_REFRESH_STOP.set()
    if FILTERS_REFRESH_TASK:
        await FILTERS_REFRESH_TASK

def normalize_format_token(value: str) -> str:
    if not value:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in FORMAT_CODE_TO_VALUE:
        return FORMAT_CODE_TO_VALUE[upper]
    lower = raw.lower()
    if "dolby" in lower and "cinema" in lower:
        return "dci"
    if "dolby" in lower and "vision" in lower:
        return "dci"
    if "dolby" in lower and "atmos" in lower:
        return "dat"
    if "screenx" in lower:
        return "scx"
    if "onyx" in lower:
        return "onx"
    if "excellence" in lower or "cst" in lower:
        return "excellence"
    if "4dx" in lower:
        return "4dx"
    if "4de" in lower or ("4d" in lower and "emotion" in lower):
        return "4de"
    if "imax" in lower and "3d" in lower:
        return "imax3d"
    if "imax" in lower:
        return "imax"
    if "35mm" in lower or "35 mm" in lower:
        return "35mm"
    if "3d" in lower:
        return "3d"
    if f" {lower} ".find(" ice ") != -1:
        return "ice"
    return "".join(ch.lower() for ch in raw if ch.isalnum())

def extract_format_tokens(value: str) -> list[str]:
    if not value:
        return []
    raw = str(value).strip()
    if not raw:
        return []
    parts = re.split(r"[,+/|]", raw)
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        subparts = part.split()
        if len(subparts) > 1 and all(re.fullmatch(r"[A-Z0-9]{2,4}", sp) for sp in subparts):
            for sp in subparts:
                token = normalize_format_token(sp)
                if token:
                    tokens.append(token)
        else:
            token = normalize_format_token(part)
            if token:
                tokens.append(token)
    lower = raw.lower()
    if "dolby cinema" in lower or "dolby vision" in lower:
        tokens.append("dci")
    if "dolby atmos" in lower or "atmos" in lower:
        tokens.append("dat")
    if "screenx" in lower:
        tokens.append("scx")
    if "onyx" in lower:
        tokens.append("onx")
    if "4dx" in lower:
        tokens.append("4dx")
    if "4de" in lower or ("4d" in lower and "emotion" in lower):
        tokens.append("4de")
    if "imax" in lower and "3d" in lower:
        tokens.append("imax3d")
    elif "imax" in lower:
        tokens.append("imax")
    if "35mm" in lower or "35 mm" in lower:
        tokens.append("35mm")
    if "3d" in lower:
        tokens.append("3d")
    if f" {lower} ".find(" ice ") != -1:
        tokens.append("ice")
    if "excellence" in lower or "cst" in lower:
        tokens.append("excellence")
    unique = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return unique

def extract_session_format_tokens(session: dict) -> list[str]:
    tokens = []
    for key in ("format", "exp", "com"):
        tokens.extend(extract_format_tokens(session.get(key, "")))
    unique = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return unique

def map_showtime_formats(value: str) -> str:
    if not value:
        return value
    parts = [v for v in value.split(",") if v]
    if not parts:
        return ""
    mapped = []
    for part in parts:
        norm = normalize_format_token(part)
        if not norm or norm == "all":
            continue
        mapped.append(FORMAT_VALUE_TO_CODE.get(norm, part))
    return ",".join(mapped)

def filter_showtimes_by_formats(showtimes: list[dict], selected_formats: list[str]) -> list[dict]:
    if not selected_formats:
        return showtimes
    return [
        session for session in showtimes
        if all(fmt in extract_session_format_tokens(session) for fmt in selected_formats)
    ]

def build_showtime_params(mov_id: str, base_params: dict) -> dict:
    params = {
        "mov_id": mov_id,
        "selday": base_params.get("selday", "week"),
        "selcard": base_params.get("selcard", "all"),
        "seladdr": base_params.get("seladdr", "75000"),
        "selcine": base_params.get("selcine", ""),
        "selformat": "all",
        "sellang": base_params.get("sellang", "all"),
    }
    return normalize_request_params(params)

def normalize_imdb_key(value: str) -> str:
    if not value:
        return ""
    raw = str(value).strip()
    if raw.startswith("tt"):
        return raw[2:]
    return raw

def tmdb_headers() -> dict:
    headers = {"accept": "application/json"}
    if TMDB_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_ACCESS_TOKEN}"
    return headers

def tmdb_find_params() -> dict:
    params = {"external_source": "imdb_id"}
    if TMDB_API_KEY:
        params["api_key"] = TMDB_API_KEY
    return params

def tmdb_movie_params(extra: Optional[dict] = None) -> dict:
    params = {"language": "fr-FR", "append_to_response": "credits,videos,watch/providers"}
    if TMDB_API_KEY:
        params["api_key"] = TMDB_API_KEY
    if extra:
        params.update(extra)
    return params

def chunked(items: list, size: int) -> list[list]:
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]

async def fetch_tmdb_movie_details(client: httpx.AsyncClient, tmdb_id: str) -> Optional[dict]:
    if not tmdb_id:
        return None
    try:
        async with TMDB_SEMAPHORE:
            resp = await client.get(
                f"{TMDB_V3_BASE}/movie/{tmdb_id}",
                params=tmdb_movie_params(),
            )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None

async def fetch_tmdb_details_for_imdb(client: httpx.AsyncClient, imdb_id: str) -> dict:
    clean_id = imdb_id if imdb_id.startswith("tt") else f"tt{imdb_id}"
    try:
        async with TMDB_SEMAPHORE:
            find_resp = await client.get(
                f"{TMDB_V3_BASE}/find/{clean_id}",
                params=tmdb_find_params(),
            )
        find_data = find_resp.json() if find_resp.status_code == 200 else {}
        if not find_data.get("movie_results"):
            return {"error": "Not found"}
        tmdb_id = find_data["movie_results"][0].get("id")
        details = await fetch_tmdb_movie_details(client, str(tmdb_id))
        if not details:
            return {"error": "TMDB Error"}
        return details
    except Exception:
        return {"error": "TMDB Error"}

async def fetch_tmdb_list_items(client: httpx.AsyncClient, list_id: str) -> list[dict]:
    if not list_id:
        return []
    items = []
    page = 1
    while True:
        params = {"page": page}
        if TMDB_API_KEY:
            params["api_key"] = TMDB_API_KEY
        try:
            async with TMDB_SEMAPHORE:
                resp = await client.get(f"{TMDB_V4_BASE}/list/{list_id}", params=params)
            if resp.status_code != 200:
                return []
            payload = resp.json()
        except Exception:
            return []
        page_items = payload.get("results")
        if page_items is None:
            page_items = payload.get("items", [])
        items.extend(page_items or [])
        total_pages = payload.get("total_pages") or 1
        if page >= total_pages:
            break
        page += 1
    return items

async def fetch_tmdb_batch_from_list(client: httpx.AsyncClient, list_id: str, missing_ids: set[str]) -> dict:
    results = {}
    if not missing_ids:
        return results
    items = await fetch_tmdb_list_items(client, list_id)
    movie_items = [item for item in items if item.get("media_type", "movie") == "movie"]
    for batch in chunked(movie_items, TMDB_MAX_CONCURRENCY):
        tasks = [fetch_tmdb_movie_details(client, str(item.get("id"))) for item in batch if item.get("id")]
        if not tasks:
            continue
        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        for data in fetched:
            if isinstance(data, Exception) or not data:
                continue
            imdb_id = data.get("imdb_id")
            if not imdb_id:
                continue
            key = normalize_imdb_key(imdb_id)
            if key in missing_ids:
                results[key] = data
                missing_ids.remove(key)
        if not missing_ids:
            break
    return results

async def fetch_tmdb_batch_from_imdb(client: httpx.AsyncClient, imdb_ids: list[str]) -> dict:
    results = {}
    if not imdb_ids:
        return results
    tasks = [fetch_tmdb_details_for_imdb(client, imdb_id) for imdb_id in imdb_ids]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)
    for imdb_id, data in zip(imdb_ids, fetched):
        if isinstance(data, Exception) or not data or data.get("error"):
            continue
        results[normalize_imdb_key(imdb_id)] = data
    return results

@app.get("/api/tmdb/details")
async def get_tmdb_details(imdb_id: str):
    cache_key = normalize_imdb_key(imdb_id)
    cached = tmdb_cache.get(cache_key) or tmdb_cache.get(imdb_id)
    if cached:
        return cached
    async with httpx.AsyncClient(timeout=10.0, headers=tmdb_headers()) as client:
        data = await fetch_tmdb_details_for_imdb(client, imdb_id)
    if data and not data.get("error"):
        tmdb_cache[cache_key] = data
        save_cache()
    return data

@app.get("/api/tmdb/batch")
async def get_tmdb_batch(imdb_ids: str, list_id: str = ""):
    raw_ids = [i.strip() for i in imdb_ids.split(",") if i.strip()]
    results = {}
    missing_map = {}
    for raw_id in raw_ids:
        cache_key = normalize_imdb_key(raw_id)
        cached = tmdb_cache.get(raw_id) or tmdb_cache.get(cache_key) or tmdb_cache.get(f"tt{cache_key}")
        if cached:
            results[raw_id] = cached
        else:
            missing_map.setdefault(cache_key, []).append(raw_id)
    if not missing_map:
        return results
    target_list_id = (list_id or TMDB_LIST_ID).strip()
    async with httpx.AsyncClient(timeout=15.0, headers=tmdb_headers()) as client:
        if target_list_id:
            fetched = await fetch_tmdb_batch_from_list(client, target_list_id, set(missing_map.keys()))
            if not fetched:
                fetched = await fetch_tmdb_batch_from_imdb(client, list(missing_map.keys()))
        else:
            fetched = await fetch_tmdb_batch_from_imdb(client, list(missing_map.keys()))
    if fetched:
        tmdb_cache.update(fetched)
        save_cache()
    for cache_key, data in fetched.items():
        for raw_id in missing_map.get(cache_key, []):
            results[raw_id] = data
    return results

async def fetch_merged_data(endpoint: str, params: dict, multi_param_names: list[str], mode: str = "any"):
    """Effectue plusieurs requêtes récursives pour gérer les multi-filtres sur plusieurs paramètres."""
    
    # Trouver le premier paramètre multi-valeurs restant
    target_param = None
    for name in multi_param_names:
        val = params.get(name, "")
        if val and "," in val:
            target_param = name
            break
            
    if not target_param:
        # Fin de la récursion ou pas de multi-valeurs, requête simple
        clean_params = normalize_request_params(params)
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"{BASE_URL}/{endpoint}"
            print(f"Fetching: {url} with {clean_params}")
            try:
                resp = await client.get(url, params=clean_params)
                if resp.status_code != 200:
                    print(f"Remote server returned {resp.status_code} for {url}")
                    return {"data": [], "showtimes": []}
                
                # Vérifier si le contenu n'est pas vide et ressemble à du JSON
                content = resp.text.strip()
                if not content or not (content.startswith('{') or content.startswith('[')):
                    print(f"Non-JSON or empty response from {endpoint}")
                    return {"data": [], "showtimes": []}
                    
                return resp.json()
            except json.JSONDecodeError:
                print(f"Invalid JSON received from {endpoint}")
                return {"data": [], "showtimes": []}
            except Exception as e:
                print(f"Error fetching {endpoint}: {e}")
                return {"data": [], "showtimes": []}
    
    # On décompose sur target_param et on appelle récursivement pour les autres
    values = [v for v in params[target_param].split(',') if v]
    
    # Sécurité : si trop de valeurs, on limite pour ne pas faire planter le serveur distant
    if len(values) > 15:
        values = values[:15]
        print(f"Warning: too many multi-values for {target_param}, limiting to 15")

    tasks = []
    
    # Copie de la liste pour ne pas modifier l'originale dans la récursion
    remaining_params = [p for p in multi_param_names if p != target_param]
    
    for val in values:
        p = params.copy()
        p[target_param] = val
        # Appel récursif pour gérer les autres paramètres multi-valeurs
        tasks.append(fetch_merged_data(endpoint, p, remaining_params, mode))
        # Petite pause pour ne pas saturer le serveur distant
        await asyncio.sleep(0.05)
        
    responses = await asyncio.gather(*tasks)
        
    first_json = None
    key = "data" if "movies" in endpoint else "showtimes"

    if mode == "all":
        # Intersection (utilisé pour les formats)
        counts = {}
        items_by_id = {}
        required = len(values)
        for js in responses:
            try:
                if first_json is None: first_json = js
                seen_ids = set()
                for item in js.get(key, []):
                    item_id = str(item.get("id") or (str(item.get("tid", "")) + str(item.get("start", ""))))
                    if item_id in seen_ids: continue
                    seen_ids.add(item_id)
                    counts[item_id] = counts.get(item_id, 0) + 1
                    items_by_id[item_id] = item
            except Exception: continue
        if first_json is None: return {key: []}
        result = first_json.copy()
        result[key] = [items_by_id[item_id] for item_id, count in counts.items() if count == required]
        return result
    else:
        # Union (utilisé pour les cinémas et localisations)
        merged_data = {}
        for js in responses:
            try:
                if first_json is None: first_json = js
                for item in js.get(key, []):
                    item_id = str(item.get("id") or (str(item.get("tid", "")) + str(item.get("start", ""))))
                    merged_data[item_id] = item
            except Exception: continue
        if first_json is None: return {key: []}
        result = first_json.copy()
        result[key] = list(merged_data.values())
        return result

async def fetch_movie_showtimes(
    client: httpx.AsyncClient,
    mov_id: str,
    base_params: dict,
    selected_formats: list[str],
) -> list[dict]:
    show_params = build_showtime_params(mov_id, base_params)
    cache_key = build_showtime_cache_key(mov_id, show_params)
    showtimes = get_cached_showtimes(cache_key)
    if showtimes is None:
        try:
            async with SHOWTIMES_SEMAPHORE:
                resp = await client.get(f"{BASE_URL}/get_pcishowtimes.php", params=show_params)
            data = resp.json()
            showtimes = data.get("showtimes", [])
            set_cached_showtimes(cache_key, showtimes)
        except Exception:
            return []
    return filter_showtimes_by_formats(showtimes, selected_formats)

async def movie_matches_formats(client: httpx.AsyncClient, movie: dict, base_params: dict, selected_formats: list[str]) -> bool:
    mov_id = movie.get("id")
    if not mov_id:
        return False
    showtimes = await fetch_movie_showtimes(client, mov_id, base_params, selected_formats)
    return bool(showtimes)

async def filter_movies_by_showtimes(movies: list[dict], base_params: dict, selected_formats: list[str]) -> list[dict]:
    if not movies or not selected_formats:
        return movies
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [
            movie_matches_formats(client, movie, base_params, selected_formats)
            for movie in movies
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    filtered = []
    for movie, matched in zip(movies, results):
        if matched is True:
            filtered.append(movie)
    return filtered

@app.get("/api/movies")
async def get_movies(
    selday: str = "week", selcard: str = "all", selformat: str = "all",
    seladdr: str = "75000", sellang: str = "all", selgenre: str = "all",
    selcine: str = "", search: str = "", selevent: str = ""
):
    selected_formats = [t for t in extract_format_tokens(selformat) if t and t != "all"]
    seldayid = get_seldayid(selday)
    params = {
        "selday": selday, "seldayid": str(seldayid),
        "selcard": selcard, "selformat": selformat,
        "seladdr": seladdr, "sellang": sellang, "selgenre": "all",
        "selcine": selcine, "init": "true", "selevent": selevent
    }
    
    # Fusion des résultats. 
    # Note: On utilise 'any' par défaut, le filtrage par format sera affiné ensuite si besoin.
    data = await fetch_merged_data("get_pcimovies.php", params, ["selcine", "selformat"], mode="any")
        
    if len(selected_formats) > 1 and isinstance(data, dict) and "data" in data:
        data["data"] = await filter_movies_by_showtimes(data.get("data", []), params, selected_formats)
    return data

@app.get("/api/showtimes")
async def get_showtimes(
    mov_id: str, selday: str = "week", selcard: str = "all", seladdr: str = "75000",
    selcine: str = "", selformat: str = "all", sellang: str = "all"
):
    selected_formats = [t for t in extract_format_tokens(selformat) if t != "all"]
    base_params = {
        "mov_id": mov_id,
        "selday": selday,
        "selcard": selcard,
        "seladdr": seladdr,
        "selcine": selcine,
        "sellang": sellang,
    }
    
    # On utilise fetch_merged_data pour les cinémas ET les localisations car PCishowtimes ne supporte pas les virgules
    data = await fetch_merged_data("get_pcishowtimes.php", base_params, ["selcine", "seladdr"], mode="any")
    showtimes = data.get("showtimes", [])
    
    # Filtrage manuel des formats sur le résultat fusionné
    if selected_formats:
        showtimes = filter_showtimes_by_formats(showtimes, selected_formats)
        
    return {"showtimes": showtimes}

@app.get("/api/filters")
async def get_filters():
    return await refresh_filters(force=False)

@app.get("/api/theatre")
async def get_theatre(theatre_id: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/get_pcitheatre.php", params={"theatre_id": theatre_id})
        return resp.json()

@app.get("/api/geocode")
async def geocode(query: str):
    resolved = resolve_location_query(query)
    if not resolved:
        raise HTTPException(status_code=404, detail="Location not found")
    return resolved

@app.get("/api/cinemas/nearby")
async def get_cinemas_nearby(lat: float, lon: float, radius_km: float = 10.0):
    if radius_km < 0:
        raise HTTPException(status_code=400, detail="radius_km must be >= 0")
    filters = await refresh_filters(force=False)
    cinemas = filters.get("cinemas", [])
    results = []
    for cinema in cinemas:
        coords = cinema_coords_from_cinema(cinema)
        if not coords:
            continue
        distance = haversine_km(lat, lon, coords[0], coords[1])
        if distance <= radius_km:
            results.append({
                **cinema,
                "lat": coords[0],
                "lon": coords[1],
                "distance_km": round(distance, 2),
            })
    results.sort(key=lambda item: item["distance_km"])
    return {"cinemas": results}

# =====================================================
# LETTERBOXD INTEGRATION
# =====================================================

LETTERBOXD_BASE = "https://letterboxd.com"
LETTERBOXD_SEMAPHORE = asyncio.Semaphore(2)

def extract_film_slugs_from_html(html: str) -> list[str]:
    """Extrait les slugs de films depuis le HTML Letterboxd."""
    slugs = []
    # Pattern pour les posters de films
    for match in re.finditer(r'data-film-slug="([^"]+)"', html):
        slug = match.group(1)
        if slug and slug not in slugs:
            slugs.append(slug)
    # Pattern alternatif pour les liens de films
    for match in re.finditer(r'/film/([a-z0-9-]+)/?["\']', html):
        slug = match.group(1)
        if slug and slug not in slugs:
            slugs.append(slug)
    return slugs

async def fetch_letterboxd_page(client: httpx.AsyncClient, url: str) -> str:
    """Fetch une page Letterboxd avec gestion de rate limiting."""
    async with LETTERBOXD_SEMAPHORE:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            resp = await client.get(url, headers=headers, follow_redirects=True)
            if resp.status_code == 200:
                return resp.text
            return ""
        except Exception:
            return ""

async def scrape_letterboxd_list(username: str, list_type: str, max_pages: int = 500) -> list[str]:
    """
    Scrape une liste Letterboxd (watchlist ou films).
    list_type: 'watchlist' ou 'films'
    """
    all_slugs = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for page in range(1, max_pages + 1):
            if page == 1:
                url = f"{LETTERBOXD_BASE}/{username}/{list_type}/"
            else:
                url = f"{LETTERBOXD_BASE}/{username}/{list_type}/page/{page}/"

            html = await fetch_letterboxd_page(client, url)
            if not html:
                break

            slugs = extract_film_slugs_from_html(html)
            if not slugs:
                break

            all_slugs.extend(slugs)

            # Check if there's a next page
            if f'/page/{page + 1}/' not in html and page > 1:
                break

    return list(dict.fromkeys(all_slugs))  # Dedupe while preserving order

@app.get("/api/letterboxd/watchlist")
async def get_letterboxd_watchlist(username: str):
    """Récupère la watchlist d'un utilisateur Letterboxd."""
    if not username or not username.strip():
        raise HTTPException(status_code=400, detail="Username required")

    username = username.strip().lower()
    films = await scrape_letterboxd_list(username, "watchlist")
    return {"films": films, "count": len(films)}

@app.get("/api/letterboxd/films")
async def get_letterboxd_films(username: str):
    """Récupère les films vus d'un utilisateur Letterboxd."""
    if not username or not username.strip():
        raise HTTPException(status_code=400, detail="Username required")

    username = username.strip().lower()
    films = await scrape_letterboxd_list(username, "films")
    return {"films": films, "count": len(films)}

# =====================================================
# AUTHENTICATION ENDPOINTS
# =====================================================

@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    """Enregistre un nouvel utilisateur."""
    user = create_user(
        email=user_data.email,
        username=user_data.username,
        password=user_data.password
    )

    access_token = create_access_token(user["email"], user["username"])

    return Token(
        access_token=access_token,
        user={
            "email": user["email"],
            "username": user["username"],
            "preferences": user.get("preferences", {}),
            "letterboxd": user.get("letterboxd", None),
            "created_at": user.get("created_at")
        }
    )

@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Authentifie un utilisateur et retourne un token JWT."""
    user = authenticate_user(credentials.email, credentials.password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    access_token = create_access_token(user["email"], user["username"])

    return Token(
        access_token=access_token,
        user={
            "email": user["email"],
            "username": user["username"],
            "preferences": user.get("preferences", {}),
            "letterboxd": user.get("letterboxd", None),
            "created_at": user.get("created_at")
        }
    )

@app.get("/api/auth/me", response_model=UserProfile)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Récupère les informations de l'utilisateur courant."""
    return UserProfile(**current_user)

@app.put("/api/auth/preferences", response_model=UserPreferences)
async def update_preferences(
    preferences: UserPreferences,
    current_user: dict = Depends(get_current_user)
):
    """Met à jour les préférences de l'utilisateur."""
    updated_prefs = update_user_preferences(
        current_user["email"],
        preferences.dict()
    )
    return UserPreferences(**updated_prefs)

@app.post("/api/auth/letterboxd/sync")
async def sync_letterboxd(
    username: str,
    current_user: dict = Depends(get_current_user)
):
    """Synchronise les données Letterboxd de l'utilisateur (requiert authentification)."""
    if not username or not username.strip():
        raise HTTPException(status_code=400, detail="Letterboxd username required")

    username = username.strip().lower()

    # Scrape les données Letterboxd
    watchlist_films = await scrape_letterboxd_list(username, "watchlist")
    watched_films = await scrape_letterboxd_list(username, "films")

    # Créer l'objet de données Letterboxd
    letterboxd_data = {
        "username": username,
        "watchlist": watchlist_films,
        "watched": watched_films,
        "lastUpdated": datetime.utcnow().isoformat()
    }

    # Sauvegarder dans le profil utilisateur
    update_user_letterboxd(current_user["email"], letterboxd_data)

    return {
        "success": True,
        "username": username,
        "watchlistCount": len(watchlist_films),
        "watchedCount": len(watched_films),
        "lastUpdated": letterboxd_data["lastUpdated"]
    }

@app.get("/api/auth/letterboxd/data")
async def get_letterboxd_data(current_user: dict = Depends(get_current_user)):
    """Récupère les données Letterboxd de l'utilisateur (depuis le cache du profil)."""
    letterboxd_data = get_user_letterboxd(current_user["email"])

    if not letterboxd_data:
        return {
            "username": None,
            "watchlist": [],
            "watched": [],
            "lastUpdated": None
        }

    return letterboxd_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

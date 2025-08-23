# Country_config.py
# Static country→market mapping with ISO codes, aliases, and fuzzy fallback.
# Responsibility: Given free text, find a valid country (code/full/alias/typo) and map it to Market A/B/C.
# If not able to map, return (None, None, None).

from typing import Dict, Set, Tuple, Optional
import re
from difflib import get_close_matches

# ---------- Canonical sets ----------
MARKET_A: Set[str] = {
    "australia","austria","belgium","canada","denmark","finland","france","germany",
    "iceland","ireland","italy","japan","luxembourg","netherlands","new zealand",
    "norway","portugal","spain","sweden","switzerland","united kingdom","united states",
}

MARKET_B: Set[str] = {
    "bahrain","barbados","brazil","cayman islands","chile","china","colombia","cyprus",
    "czechia","estonia","greece","hong kong","indonesia","israel","jamaica","korea",
    "kuwait","latvia","lithuania","malaysia","malta","mexico","north macedonia","oman",
    "philippines","poland","puerto rico","qatar","saudi arabia","senegal","singapore",
    "slovakia","slovenia","south africa","taiwan","thailand","united arab emirates","uruguay",
}

# Common “Market C” (others). Expand as needed.
MARKET_C_KNOWN: Set[str] = {
    "india", "vietnam", "pakistan", "bangladesh", "nepal", "sri lanka", "argentina",
    "peru", "nigeria", "kenya", "morocco", "algeria", "tunisia", "turkey",
}

MARKET_RATE: Dict[str, int] = {"A": 163, "B": 116, "C": 70}

# ---------- ISO codes ----------
# NOTE: ISO2 matching is restricted to UPPERCASE tokens to avoid false positives like the word "in".
ISO2_TO_NAME: Dict[str, str] = {
    # A
    "AU":"australia","AT":"austria","BE":"belgium","CA":"canada","DK":"denmark","FI":"finland",
    "FR":"france","DE":"germany","IS":"iceland","IE":"ireland","IT":"italy","JP":"japan",
    "LU":"luxembourg","NL":"netherlands","NZ":"new zealand","NO":"norway","PT":"portugal",
    "ES":"spain","SE":"sweden","CH":"switzerland","GB":"united kingdom","UK":"united kingdom","US":"united states",
    # B
    "BH":"bahrain","BB":"barbados","BR":"brazil","KY":"cayman islands","CL":"chile","CN":"china",
    "CO":"colombia","CY":"cyprus","CZ":"czechia","EE":"estonia","GR":"greece","HK":"hong kong",
    "ID":"indonesia","IL":"israel","JM":"jamaica","KR":"korea","KW":"kuwait","LV":"latvia",
    "LT":"lithuania","MY":"malaysia","MT":"malta","MX":"mexico","MK":"north macedonia","OM":"oman",
    "PH":"philippines","PL":"poland","PR":"puerto rico","QA":"qatar","SA":"saudi arabia","SN":"senegal",
    "SG":"singapore","SK":"slovakia","SI":"slovenia","ZA":"south africa","TW":"taiwan","TH":"thailand",
    "AE":"united arab emirates","UY":"uruguay",
    # C (selected common; add more if you like)
    "IN":"india","VN":"vietnam","PK":"pakistan","BD":"bangladesh","NP":"nepal","LK":"sri lanka",
    "TR":"turkey","NG":"nigeria","KE":"kenya","AR":"argentina","PE":"peru","MA":"morocco",
    "DZ":"algeria","TN":"tunisia",
}

ISO3_TO_NAME: Dict[str, str] = {
    # A
    "AUS":"australia","AUT":"austria","BEL":"belgium","CAN":"canada","DNK":"denmark","FIN":"finland",
    "FRA":"france","DEU":"germany","ISL":"iceland","IRL":"ireland","ITA":"italy","JPN":"japan",
    "LUX":"luxembourg","NLD":"netherlands","NZL":"new zealand","NOR":"norway","PRT":"portugal",
    "ESP":"spain","SWE":"sweden","CHE":"switzerland","GBR":"united kingdom","USA":"united states",
    # B
    "BHR":"bahrain","BRB":"barbados","BRA":"brazil","CYM":"cayman islands","CHL":"chile","CHN":"china",
    "COL":"colombia","CYP":"cyprus","CZE":"czechia","EST":"estonia","GRC":"greece","HKG":"hong kong",
    "IDN":"indonesia","ISR":"israel","JAM":"jamaica","KOR":"korea","KWT":"kuwait","LVA":"latvia",
    "LTU":"lithuania","MYS":"malaysia","MLT":"malta","MEX":"mexico","MKD":"north macedonia",
    "OMN":"oman","PHL":"philippines","POL":"poland","PRI":"puerto rico","QAT":"qatar","SAU":"saudi arabia",
    "SEN":"senegal","SGP":"singapore","SVK":"slovakia","SVN":"slovenia","ZAF":"south africa",
    "TWN":"taiwan","THA":"thailand","ARE":"united arab emirates","URY":"uruguay",
    # C (selected common)
    "IND":"india","VNM":"vietnam","PAK":"pakistan","BGD":"bangladesh","NPL":"nepal","LKA":"sri lanka",
    "TUR":"turkey","NGA":"nigeria","KEN":"kenya","ARG":"argentina","PER":"peru","MAR":"morocco",
    "DZA":"algeria","TUN":"tunisia",
}

# ---------- Aliases / colloquials / typos (map to canonical) ----------
ALIASES: Dict[str, str] = {
    # US/UK/UAE
    "usa":"united states","u.s.a":"united states","u.s.a.":"united states","us":"united states","u.s.":"united states",
    "america":"united states","united states of america":"united states",
    "uk":"united kingdom","u.k.":"united kingdom","u.k":"united kingdom","great britain":"united kingdom",
    "britain":"united kingdom","england":"united kingdom",
    "uae":"united arab emirates","u.a.e.":"united arab emirates","u.a.e":"united arab emirates",
    # Korea
    "south korea":"korea","republic of korea":"korea","rok":"korea",
    # Czechia
    "czech republic":"czechia",
    # North Macedonia
    "n. macedonia":"north macedonia","macedonia":"north macedonia",
    # Hong Kong
    "hong kong sar":"hong kong","hk":"hong kong",
    # Common C
    "bharat":"india","hindustan":"india",
}

def _classify_market(canon: str) -> Tuple[str, int]:
    if canon in MARKET_A: return "A", MARKET_RATE["A"]
    if canon in MARKET_B: return "B", MARKET_RATE["B"]
    if canon in MARKET_C_KNOWN: return "C", MARKET_RATE["C"]
    # Unknown → let caller treat as unmapped
    raise KeyError

def resolve_market_from_text(text: str, fuzzy: bool = True, cutoff: float = 0.86) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Resolve a country from free text (code / full name / alias / minor typo) and map to market.
    Returns (country_canonical, market_letter, hourly_rate). If not mapped → (None, None, None).
    Rules:
      - ISO2 codes only when UPPERCASE (to avoid matching words like 'in').
      - ISO3 codes case-insensitive.
      - Full names / aliases via word-boundary search.
      - Fuzzy match last (minor typos).
      - Market C applies to known-countries not present in A/B (e.g., India).
    """
    if not text:
        return None, None, None

    # 1) ISO2 (UPPERCASE tokens only)
    for code, canon in ISO2_TO_NAME.items():
        if re.search(rf"\b{code}\b", text):
            try:
                market, rate = _classify_market(canon)
                return canon, market, rate
            except KeyError:
                return canon, "C", MARKET_RATE["C"]

    low = text.lower()

    # 2) ISO3 (case-insensitive)
    for code, canon in ISO3_TO_NAME.items():
        if re.search(rf"\b{code.lower()}\b", low):
            try:
                market, rate = _classify_market(canon)
                return canon, market, rate
            except KeyError:
                return canon, "C", MARKET_RATE["C"]

    # 3) Aliases / canonical names (word boundaries; prefer longer first)
    names = list(MARKET_A | MARKET_B | MARKET_C_KNOWN)
    for term in sorted(set(list(ALIASES.keys()) + names), key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", low):
            canon = ALIASES.get(term, term)
            try:
                market, rate = _classify_market(canon)
                return canon, market, rate
            except KeyError:
                return canon, "C", MARKET_RATE["C"]

    # 4) Fuzzy fallback for small typos (check aliases+names)
    if fuzzy:
        candidates = list(set(list(ALIASES.keys()) + names))
        # scan 1–3 word windows
        windows = re.findall(r"[a-z]+(?:\s+[a-z]+){0,2}", low)
        for w in sorted(set(windows), key=len, reverse=True):
            hit = get_close_matches(w, candidates, n=1, cutoff=cutoff)
            if hit:
                canon = ALIASES.get(hit[0], hit[0])
                try:
                    market, rate = _classify_market(canon)
                    return canon, market, rate
                except KeyError:
                    return canon, "C", MARKET_RATE["C"]

    return None, None, None

"""
Science MCP Server — поиск научных статей для ИИ-ассистента.

Инструменты:
  search_all                — общий поиск: OpenAlex + КиберЛенинка + Semantic Scholar, с дедупликацией
  cyberleninka_search       — российские статьи в открытом доступе (КиберЛенинка)
  cyberleninka_get_article  — полный текст статьи КиберЛенинки по ссылке
  openalex_search           — мировые статьи с DOI (включая российские журналы), фильтры по годам/языку
  article_by_doi            — метаданные статьи по DOI (Crossref + OpenAlex): точные даты онлайн/печать
  europepmc_search / europepmc_full_text — Europe PMC (биомедицина, полные тексты OA, раздел методов)
  semantic_scholar_search   — Semantic Scholar (аннотации, цитирования)
  google_scholar_search     — Google Академия через SerpAPI (нужен SERPAPI_KEY); без ключа — замена на OpenAlex+S2

eLIBRARY.ru не подключается: у него нет открытого API, автоматический доступ запрещён правилами сайта.

Переменные окружения:
  MCP_PATH            — секретный путь эндпоинта (как у patent-mcp)
  SERPAPI_KEY         — ключ serpapi.com для Google Академии (необязательно)
  SEMANTIC_SCHOLAR_KEY— ключ Semantic Scholar API (необязательно, снижает ошибки 429)
  OPENALEX_MAILTO     — e-mail для «вежливого пула» OpenAlex (необязательно)
  HTTP_TIMEOUT        — таймаут внешних запросов, сек (по умолчанию 25)
"""

from __future__ import annotations

import asyncio
import contextlib
import html as html_lib
import logging
import os
import re
from typing import Any
from urllib.parse import quote, urljoin

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

MCP_PATH = "/" + os.getenv("MCP_PATH", "mcp").strip().strip("/")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
S2_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY", "").strip()
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))

UA = {
    "User-Agent": "science-mcp/1.0 (research assistant; +https://github.com)",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.6",
}
BROWSER_UA = {
    **UA,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0 Safari/537.36",
}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("science-mcp")


class SourceError(Exception):
    pass


def _clean(text: Any) -> str:
    if not text:
        return ""
    text = html_lib.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _clamp(v: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return lo


def _err(source: str, msg: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "source": source, "error": msg, **extra}


def _norm_doi(doi: str) -> str:
    d = (doi or "").strip()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d, flags=re.I)
    return d.removeprefix("doi:").strip()


async def _get(url: str, *, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as c:
        return await c.get(url, params=params, headers={**UA, **(headers or {})})


def _status_check(r: httpx.Response, name: str) -> None:
    if r.status_code == 429:
        raise SourceError(f"{name}: превышен лимит запросов (429), повторите через минуту")
    if r.status_code in (401, 403):
        raise SourceError(f"{name}: доступ запрещён (HTTP {r.status_code}) — возможна блокировка IP или нужен ключ")
    if r.status_code >= 400:
        raise SourceError(f"{name}: HTTP {r.status_code}")


# ──────────────────────────── OpenAlex ────────────────────────────

def _oa_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            pos.append((i, word))
    return " ".join(w for _, w in sorted(pos))


def _oa_work(w: dict) -> dict[str, Any]:
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    return {
        "title": _clean(w.get("display_name") or w.get("title")),
        "authors": [_clean((a.get("author") or {}).get("display_name")) for a in (w.get("authorships") or [])][:15],
        "institutions": sorted({_clean(i.get("display_name"))
                                for a in (w.get("authorships") or []) for i in (a.get("institutions") or [])
                                if i.get("display_name")})[:10],
        "journal": _clean(src.get("display_name")),
        "year": w.get("publication_year"),
        "publication_date": w.get("publication_date"),
        "doi": _norm_doi(w.get("doi") or ""),
        "language": w.get("language"),
        "cited_by": w.get("cited_by_count"),
        "abstract": _oa_abstract(w.get("abstract_inverted_index"))[:3000],
        "open_access_url": (w.get("open_access") or {}).get("oa_url") or loc.get("landing_page_url") or "",
        "openalex_id": w.get("id"),
        "source_db": "OpenAlex",
    }


async def _openalex(query: str, limit: int, year_from: int | None = None, year_to: int | None = None,
                    language: str = "") -> tuple[int, list[dict]]:
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{int(year_from)}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{int(year_to)}-12-31")
    if language:
        filters.append(f"language:{language}")
    params: dict[str, Any] = {"search": query, "per-page": limit}
    if filters:
        params["filter"] = ",".join(filters)
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    r = await _get("https://api.openalex.org/works", params=params)
    _status_check(r, "OpenAlex")
    data = r.json()
    return (data.get("meta") or {}).get("count", 0), [_oa_work(w) for w in data.get("results", [])]


# ──────────────────────────── КиберЛенинка ────────────────────────────

CL_BASE = "https://cyberleninka.ru"


async def _cyberleninka(query: str, limit: int, year_from: int | None = None) -> tuple[int, list[dict]]:
    payload = {"mode": "articles", "q": query, "size": limit, "from": 0}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as c:
        r = await c.post(f"{CL_BASE}/api/search", json=payload,
                         headers={**BROWSER_UA, "Content-Type": "application/json",
                                  "Referer": f"{CL_BASE}/search?q={quote(query)}"})
    _status_check(r, "КиберЛенинка")
    try:
        data = r.json()
    except ValueError as e:
        raise SourceError("КиберЛенинка вернула не-JSON (вероятно, защита от ботов)") from e
    out = []
    for a in data.get("articles", []) or []:
        year = a.get("year")
        if year_from and isinstance(year, int) and year < year_from:
            continue
        link = a.get("link") or ""
        out.append({
            "title": _clean(a.get("name")),
            "authors": [_clean(x) for x in (a.get("authors") or [])][:15],
            "journal": _clean(a.get("journal")),
            "year": year,
            "abstract": _clean(a.get("annotation"))[:3000],
            "url": urljoin(CL_BASE, link) if link else "",
            "source_db": "КиберЛенинка",
        })
    return data.get("found", len(out)), out


async def _cyberleninka_article(url: str, max_chars: int) -> dict[str, Any]:
    if not url.startswith("http"):
        url = urljoin(CL_BASE, url)
    if "cyberleninka.ru/article/" not in url:
        raise SourceError("Ожидается ссылка вида https://cyberleninka.ru/article/n/...")
    r = await _get(url, headers=BROWSER_UA)
    _status_check(r, "КиберЛенинка")
    soup = BeautifulSoup(r.text, "html.parser")

    def first(*selectors: str) -> str:
        for s in selectors:
            el = soup.select_one(s)
            if el:
                t = _clean(el.get_text(" "))
                if t:
                    return t
        return ""

    def meta(name: str) -> list[str]:
        return [_clean(m.get("content")) for m in soup.find_all("meta", attrs={"name": name}) if m.get("content")]

    text = first("div.ocr[itemprop=articleBody]", "div.ocr", "[itemprop=articleBody]")
    return {
        "ok": True,
        "source": "КиберЛенинка",
        "url": url,
        "title": (meta("citation_title") or [first("h1 i", "h1")])[0],
        "authors": meta("citation_author")[:20],
        "journal": (meta("citation_journal_title") or [""])[0],
        "year": (meta("citation_publication_date") or meta("citation_date") or [""])[0],
        "doi": (meta("citation_doi") or [""])[0],
        "pdf_url": (meta("citation_pdf_url") or [""])[0],
        "abstract": first("div.abstract p[itemprop=description]", "[itemprop=description]", "div.abstract"),
        "keywords": [_clean(k.get_text()) for k in soup.select("i[itemprop=keywords] span")][:20],
        "text_length": len(text),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


# ──────────────────────────── Crossref ────────────────────────────

def _date_parts(obj: dict | None) -> str:
    try:
        p = (obj or {}).get("date-parts", [[]])[0]
        return "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(p))
    except (TypeError, IndexError, ValueError):
        return ""


async def _crossref(doi: str) -> dict[str, Any]:
    r = await _get(f"https://api.crossref.org/works/{quote(doi, safe='/')}")
    if r.status_code == 404:
        raise SourceError("Crossref: DOI не найден")
    _status_check(r, "Crossref")
    m = r.json().get("message", {})
    return {
        "title": _clean((m.get("title") or [""])[0]),
        "authors": [_clean(f"{a.get('family', '')} {a.get('given', '')}") for a in (m.get("author") or [])][:20],
        "journal": _clean((m.get("container-title") or [""])[0]),
        "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page"),
        "published_online": _date_parts(m.get("published-online")),
        "published_print": _date_parts(m.get("published-print")),
        "issued": _date_parts(m.get("issued")),
        "doi_registered": ((m.get("created") or {}).get("date-time") or "")[:10],
        "publisher": m.get("publisher"),
        "url": m.get("URL"),
        "abstract": _clean(m.get("abstract"))[:3000],
        "license": [l.get("URL") for l in (m.get("license") or [])][:3],
    }


# ──────────────────────────── Semantic Scholar ────────────────────────────

S2_FIELDS = "title,year,authors,venue,abstract,externalIds,url,citationCount,publicationDate,openAccessPdf"


async def _semantic(query: str, limit: int, year_from: int | None = None) -> tuple[int, list[dict]]:
    params: dict[str, Any] = {"query": query, "limit": limit, "fields": S2_FIELDS}
    if year_from:
        params["year"] = f"{int(year_from)}-"
    headers = {"x-api-key": S2_KEY} if S2_KEY else None
    r = await _get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=headers)
    _status_check(r, "Semantic Scholar")
    data = r.json()
    out = []
    for p in data.get("data", []) or []:
        ext = p.get("externalIds") or {}
        out.append({
            "title": _clean(p.get("title")),
            "authors": [_clean(a.get("name")) for a in (p.get("authors") or [])][:15],
            "journal": _clean(p.get("venue")),
            "year": p.get("year"),
            "publication_date": p.get("publicationDate"),
            "doi": _norm_doi(ext.get("DOI") or ""),
            "pmid": ext.get("PubMed"),
            "cited_by": p.get("citationCount"),
            "abstract": _clean(p.get("abstract"))[:3000],
            "url": p.get("url"),
            "open_access_url": (p.get("openAccessPdf") or {}).get("url") or "",
            "source_db": "Semantic Scholar",
        })
    return data.get("total", len(out)), out


# ──────────────────────────── Europe PMC ────────────────────────────

async def _europepmc(query: str, limit: int, year_from: int | None = None) -> tuple[int, list[dict]]:
    q = query if not year_from else f"({query}) AND PUB_YEAR:[{int(year_from)} TO 3000]"
    r = await _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                   params={"query": q, "format": "json", "pageSize": limit, "resultType": "core"})
    _status_check(r, "Europe PMC")
    data = r.json()
    out = []
    for a in (data.get("resultList") or {}).get("result", []) or []:
        out.append({
            "title": _clean(a.get("title")),
            "authors": [x.strip() for x in (a.get("authorString") or "").split(",") if x.strip()][:15],
            "journal": _clean(((a.get("journalInfo") or {}).get("journal") or {}).get("title") or a.get("journalTitle")),
            "year": int(a["pubYear"]) if str(a.get("pubYear", "")).isdigit() else a.get("pubYear"),
            "publication_date": a.get("firstPublicationDate"),
            "doi": _norm_doi(a.get("doi") or ""),
            "pmid": a.get("pmid"),
            "pmcid": a.get("pmcid"),
            "cited_by": a.get("citedByCount"),
            "abstract": _clean(a.get("abstractText"))[:3000],
            "open_access": a.get("isOpenAccess") == "Y",
            "url": f"https://europepmc.org/article/{a.get('source', 'MED')}/{a.get('id')}" if a.get("id") else "",
            "source_db": "Europe PMC",
        })
    return data.get("hitCount", len(out)), out


async def _europepmc_fulltext_methods(pmcid: str, max_chars: int) -> dict[str, Any]:
    r = await _get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
    if r.status_code == 404:
        raise SourceError("полный текст недоступен (не Open Access)")
    _status_check(r, "Europe PMC")
    soup = BeautifulSoup(r.text, "html.parser")
    title = _clean((soup.find("article-title") or soup.new_tag("x")).get_text(" "))
    secs = []
    for sec in soup.find_all("sec"):
        t = sec.find("title")
        secs.append((_clean(t.get_text(" ")) if t else "", _clean(sec.get_text(" "))))
    methods = " ".join(txt for ttl, txt in secs if re.search(r"method|материал|primer|pcr", ttl, re.I))
    body = _clean((soup.find("body") or soup).get_text(" "))
    return {"ok": True, "source": "Europe PMC", "pmcid": pmcid, "title": title,
            "methods": methods[:max_chars], "text": body[:max_chars], "truncated": len(body) > max_chars,
            "sections": [t for t, _ in secs if t][:40]}


# ──────────────────────────── Google Scholar (SerpAPI) ────────────────────────────

async def _google_scholar(query: str, limit: int, year_from: int | None, lang: str) -> list[dict]:
    if not SERPAPI_KEY:
        raise SourceError("не задан SERPAPI_KEY (ключ serpapi.com) — прямой доступ к Google Академии с сервера блокируется капчей")
    params: dict[str, Any] = {"engine": "google_scholar", "q": query, "num": min(limit, 20),
                              "api_key": SERPAPI_KEY, "hl": lang or "ru"}
    if year_from:
        params["as_ylo"] = int(year_from)
    r = await _get("https://serpapi.com/search.json", params=params)
    _status_check(r, "SerpAPI")
    data = r.json()
    if data.get("error"):
        raise SourceError(f"SerpAPI: {data['error']}")
    out = []
    for it in data.get("organic_results", []) or []:
        pub = it.get("publication_info") or {}
        out.append({
            "title": _clean(it.get("title")),
            "publication_info": _clean(pub.get("summary")),
            "snippet": _clean(it.get("snippet")),
            "url": it.get("link") or "",
            "cited_by": ((it.get("inline_links") or {}).get("cited_by") or {}).get("total"),
            "files": [x.get("link") for x in (it.get("resources") or []) if x.get("link")][:3],
            "source_db": "Google Академия",
        })
    return out


# ──────────────────────────── MCP ────────────────────────────

mcp = FastMCP(
    name="science-ru",
    instructions=(
        "Поиск научных статей. Российские: КиберЛенинка, OpenAlex с language=ru. Зарубежные: OpenAlex, Crossref, Europe PMC, "
        "Semantic Scholar и Google Академия (если задан ключ SerpAPI). Для широкого поиска начинай с search_all; "
        "для точной даты публикации статьи — article_by_doi; полный текст российской статьи — cyberleninka_get_article. "
        "eLIBRARY.ru недоступен. Всегда указывай источник и DOI/ссылку. Для биомедицины дополнительно используй PubMed."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path=MCP_PATH,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
async def search_all(query: str, limit_per_source: int = 8, year_from: int = 0) -> dict[str, Any]:
    """Общий поиск статей сразу в OpenAlex, КиберЛенинке, Semantic Scholar и Europe PMC (параллельно), с удалением дублей по DOI/названию.

    Args:
        query: запрос (на русском или английском; для полноты сделай два вызова — RU и EN).
        limit_per_source: результатов с каждого источника (1–25).
        year_from: не раньше этого года (0 — без ограничения).
    """
    n = _clamp(limit_per_source, 1, 25)
    yf = year_from or None
    tasks = {
        "OpenAlex": _openalex(query, n, yf),
        "КиберЛенинка": _cyberleninka(query, n, yf),
        "Semantic Scholar": _semantic(query, n, yf),
        "Europe PMC": _europepmc(query, n, yf),
    }
    res = await asyncio.gather(*tasks.values(), return_exceptions=True)
    warnings, merged, seen = [], [], set()
    counts = {}
    for name, r in zip(tasks, res):
        if isinstance(r, Exception):
            warnings.append(f"{name}: {r}" if isinstance(r, SourceError) else f"{name}: {type(r).__name__}")
            continue
        total, items = r
        counts[name] = {"found": total, "returned": len(items)}
        for it in items:
            key = (it.get("doi") or "").lower() or re.sub(r"\W+", "", (it.get("title") or "").lower())[:80]
            if key and key in seen:
                continue
            seen.add(key)
            merged.append(it)
    if not merged and warnings:
        return _err("search_all", "Ни один источник не ответил", warnings=warnings)
    return {"ok": True, "counts": counts, "warnings": warnings, "count": len(merged), "results": merged}


@mcp.tool()
async def cyberleninka_search(query: str, limit: int = 10, year_from: int = 0) -> dict[str, Any]:
    """Поиск российских научных статей в открытом доступе (КиберЛенинка): название, авторы, журнал, год, аннотация, ссылка.

    Args:
        query: запрос (лучше на русском), напр. «экспрессия MUC16 рак яичника ПЦР».
        limit: 1–50. year_from: не раньше этого года (0 — без ограничения).
    """
    try:
        total, items = await _cyberleninka(query, _clamp(limit, 1, 50), year_from or None)
        return {"ok": True, "source": "КиберЛенинка", "found": total, "count": len(items), "results": items}
    except (SourceError, httpx.HTTPError) as e:
        return _err("КиберЛенинка", str(e) or type(e).__name__)
    except Exception as e:  # noqa: BLE001
        log.exception("cyberleninka_search")
        return _err("КиберЛенинка", f"Непредвиденная ошибка: {type(e).__name__}")


@mcp.tool()
async def cyberleninka_get_article(url: str, max_chars: int = 20000) -> dict[str, Any]:
    """Полный текст и метаданные статьи КиберЛенинки (ссылка из cyberleninka_search): аннотация, ключевые слова, DOI, текст.

    Args:
        url: https://cyberleninka.ru/article/n/...
        max_chars: сколько символов текста вернуть (1000–60000).
    """
    try:
        return await _cyberleninka_article(url.strip(), _clamp(max_chars, 1000, 60000))
    except (SourceError, httpx.HTTPError) as e:
        return _err("КиберЛенинка", str(e) or type(e).__name__)
    except Exception as e:  # noqa: BLE001
        log.exception("cyberleninka_get_article")
        return _err("КиберЛенинка", f"Непредвиденная ошибка: {type(e).__name__}")


@mcp.tool()
async def openalex_search(query: str, limit: int = 10, year_from: int = 0, year_to: int = 0,
                          language: str = "") -> dict[str, Any]:
    """Поиск статей в OpenAlex (≈250 млн работ, включая российские журналы с DOI): авторы, организации, журнал, дата, DOI, аннотация, цитирования.

    Args:
        query: запрос. limit: 1–50.
        year_from, year_to: границы года публикации (0 — без ограничения).
        language: код языка ISO 639-1, напр. "ru" или "en" (пусто — любой).
    """
    try:
        total, items = await _openalex(query, _clamp(limit, 1, 50), year_from or None, year_to or None,
                                       language.strip().lower())
        return {"ok": True, "source": "OpenAlex", "found": total, "count": len(items), "results": items}
    except (SourceError, httpx.HTTPError) as e:
        return _err("OpenAlex", str(e) or type(e).__name__)
    except Exception as e:  # noqa: BLE001
        log.exception("openalex_search")
        return _err("OpenAlex", f"Непредвиденная ошибка: {type(e).__name__}")


@mcp.tool()
async def article_by_doi(doi: str) -> dict[str, Any]:
    """Метаданные статьи по DOI из Crossref и OpenAlex: точные даты (онлайн, печать, регистрация DOI), выходные данные, аннотация.
    Полезно для проверки льготного периода по новизне (6 месяцев с даты раскрытия).

    Args:
        doi: например 10.17650/2313-805X-2026-13-1-62-71 или https://doi.org/...
    """
    d = _norm_doi(doi)
    if not d.startswith("10."):
        return _err("doi", "Ожидается DOI вида 10.xxxx/...")
    cr, oa = await asyncio.gather(_crossref(d), _get(f"https://api.openalex.org/works/doi:{quote(d, safe='/')}"),
                                  return_exceptions=True)
    out: dict[str, Any] = {"ok": True, "doi": d, "warnings": []}
    if isinstance(cr, Exception):
        out["warnings"].append(f"Crossref: {cr}" if isinstance(cr, SourceError) else f"Crossref: {type(cr).__name__}")
    else:
        out["crossref"] = cr
    if isinstance(oa, Exception):
        out["warnings"].append(f"OpenAlex: {type(oa).__name__}")
    elif oa.status_code == 200:
        out["openalex"] = _oa_work(oa.json())
    else:
        out["warnings"].append(f"OpenAlex: HTTP {oa.status_code}")
    if "crossref" not in out and "openalex" not in out:
        return _err("doi", "DOI не найден ни в Crossref, ни в OpenAlex", warnings=out["warnings"])
    return out


@mcp.tool()
async def europepmc_search(query: str, limit: int = 10, year_from: int = 0) -> dict[str, Any]:
    """Europe PMC — зарубежная биомедицинская база (PubMed + PMC + препринты + патенты EPO): аннотации, PMID/PMCID, флаг открытого доступа.
    Поддерживает синтаксис Europe PMC: "MUC16 AND (qPCR OR \"real-time PCR\") AND primer".

    Args:
        query: запрос (на английском). limit: 1–50. year_from: не раньше этого года.
    """
    try:
        total, items = await _europepmc(query, _clamp(limit, 1, 50), year_from or None)
        return {"ok": True, "source": "Europe PMC", "found": total, "count": len(items), "results": items}
    except (SourceError, httpx.HTTPError) as e:
        return _err("Europe PMC", str(e) or type(e).__name__)
    except Exception as e:  # noqa: BLE001
        log.exception("europepmc_search")
        return _err("Europe PMC", f"Непредвиденная ошибка: {type(e).__name__}")


@mcp.tool()
async def europepmc_full_text(pmcid: str, max_chars: int = 20000) -> dict[str, Any]:
    """Полный текст статьи открытого доступа из Europe PMC по PMCID (напр. PMC2613075); отдельно выделяет раздел методов
    (удобно искать последовательности праймеров и зондов).

    Args:
        pmcid: идентификатор вида PMC1234567. max_chars: 1000–60000.
    """
    p = pmcid.strip().upper()
    if not p.startswith("PMC"):
        p = "PMC" + re.sub(r"\D", "", p)
    try:
        return await _europepmc_fulltext_methods(p, _clamp(max_chars, 1000, 60000))
    except (SourceError, httpx.HTTPError) as e:
        return _err("Europe PMC", str(e) or type(e).__name__)
    except Exception as e:  # noqa: BLE001
        log.exception("europepmc_full_text")
        return _err("Europe PMC", f"Непредвиденная ошибка: {type(e).__name__}")


@mcp.tool()
async def semantic_scholar_search(query: str, limit: int = 10, year_from: int = 0) -> dict[str, Any]:
    """Поиск в Semantic Scholar: аннотации, цитирования, DOI, PMID, ссылки на открытые PDF. Без ключа возможны ошибки 429 — повтори позже.

    Args:
        query: запрос (лучше на английском). limit: 1–50. year_from: не раньше этого года.
    """
    try:
        total, items = await _semantic(query, _clamp(limit, 1, 50), year_from or None)
        return {"ok": True, "source": "Semantic Scholar", "found": total, "count": len(items), "results": items}
    except (SourceError, httpx.HTTPError) as e:
        return _err("Semantic Scholar", str(e) or type(e).__name__)
    except Exception as e:  # noqa: BLE001
        log.exception("semantic_scholar_search")
        return _err("Semantic Scholar", f"Непредвиденная ошибка: {type(e).__name__}")


@mcp.tool()
async def google_scholar_search(query: str, limit: int = 10, year_from: int = 0, lang: str = "ru") -> dict[str, Any]:
    """Google Академия (через SerpAPI). Если ключ SERPAPI_KEY не задан или лимит исчерпан —
    автоматически возвращает замену из OpenAlex и Semantic Scholar (поле source покажет, что сработало).

    Args:
        query: запрос. limit: 1–20. year_from: не раньше этого года. lang: язык интерфейса (ru/en).
    """
    n = _clamp(limit, 1, 20)
    try:
        items = await _google_scholar(query, n, year_from or None, lang)
        return {"ok": True, "source": "Google Академия (SerpAPI)", "count": len(items), "results": items}
    except (SourceError, httpx.HTTPError) as e:
        reason = str(e) or type(e).__name__
    except Exception as e:  # noqa: BLE001
        log.exception("google_scholar_search")
        reason = f"Непредвиденная ошибка: {type(e).__name__}"
    fb = await search_all(query, n, year_from)
    fb_items = [x for x in fb.get("results", []) if x.get("source_db") != "КиберЛенинка"] if fb.get("ok") else []
    return {"ok": bool(fb_items), "source": "Замена: OpenAlex + Semantic Scholar",
            "warnings": [f"Google Академия недоступна: {reason}"] + fb.get("warnings", []),
            "count": len(fb_items), "results": fb_items}


# ──────────────────────────── FastAPI ────────────────────────────

mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp.session_manager.run():
        log.info("Science MCP endpoint: %s | SerpAPI: %s", MCP_PATH, "задан" if SERPAPI_KEY else "нет")
        yield


app = FastAPI(title="Science MCP Server", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": "science-mcp", "status": "ok"})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "serpapi_key": bool(SERPAPI_KEY), "semantic_scholar_key": bool(S2_KEY)})


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return JSONResponse({"error": "internal_error", "detail": type(exc).__name__}, status_code=500)


app.mount("/", mcp_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("science:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

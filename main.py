"""
Patent MCP Server — MCP-коннектор для ИИ-ассистента (Claude и др.)

Инструменты:
  1. search_yandex_patents     — поиск РФ-патентов (Яндекс.Патенты → запасные источники)
  2. rospatent_search          — поиск через API Поисковой платформы Роспатента (нужен токен)
     rospatent_get_document    — полный документ из Поисковой платформы по ID
     fips_get_document         — карточка документа из открытых реестров ФИПС по номеру (без токена)
  3. search_kfu_patents        — патенты, где правообладатель — Казанский федеральный университет
  4. get_patenting_guidelines  — регламенты Роспатента (формула, описание, реферат, пошлины, подача)

Транспорт: MCP Streamable HTTP (stateless, JSON) поверх FastAPI.
Эндпоинт MCP: https://<ваш-домен>/<MCP_PATH>   (по умолчанию /mcp)

Переменные окружения:
  ROSPATENT_TOKEN   — JWT-токен API Поисковой платформы Роспатента (необязателен, но сильно расширяет поиск)
  MCP_PATH          — «секретный» путь эндпоинта, напр. mcp-7f3k9q2x (защищает от случайных посетителей)
  REGULATIONS_FILE  — путь к файлу с регламентами (по умолчанию regulations.md рядом с main.py)
  HTTP_TIMEOUT      — таймаут внешних запросов, сек (по умолчанию 25)
  LOG_LEVEL         — INFO / DEBUG
"""

from __future__ import annotations

import contextlib
import html as html_lib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

ROSPATENT_TOKEN = os.getenv("ROSPATENT_TOKEN", "").strip()
MCP_PATH = "/" + os.getenv("MCP_PATH", "mcp").strip().strip("/")
REGULATIONS_FILE = Path(
    os.getenv("REGULATIONS_FILE", str(Path(__file__).with_name("regulations.md")))
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

ROSPATENT_API = "https://searchplatform.rospatent.gov.ru/patsearch/v0.2"
ROSPATENT_WEB = "https://searchplatform.rospatent.gov.ru"
FIPS_DOC_URL = "https://www.fips.ru/registers-doc-view/fips_servlet"
YANDEX_PATENTS_URL = "https://yandex.ru/patents"
GOOGLE_PATENTS_XHR = "https://patents.google.com/xhr/query"

KFU_NAME_MARKERS = ("казанск", "федеральн")  # оба должны встретиться в имени правообладателя
KFU_DEFAULT_QUERY = "Казанский (Приволжский) федеральный университет"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.6",
}

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("patent-mcp")


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────


class SourceError(Exception):
    """Ожидаемая ошибка источника данных (недоступен, капча, нет токена и т.п.)."""


def _clean(text: Any) -> str:
    """Убирает HTML-теги и лишние пробелы."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _clamp(value: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def _dig(data: Any, *keys: Any, default: Any = None) -> Any:
    """Безопасное извлечение вложенных значений: _dig(d, 'a', 0, 'b')."""
    cur = data
    for k in keys:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return default
    return cur if cur is not None else default


def _names(items: Any) -> list[str]:
    """Список имён из [{'name': ...}] / ['...'] / '...'."""
    if not items:
        return []
    if isinstance(items, str):
        return [_clean(items)]
    out = []
    for it in items:
        if isinstance(it, dict):
            name = it.get("name") or it.get("value") or ""
        else:
            name = it
        name = _clean(name)
        if name:
            out.append(name)
    return out


def _error(source: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "source": source, "error": message, **extra}


async def _http_get(url: str, *, params: dict | None = None, headers: dict | None = None,
                    follow_redirects: bool = True) -> httpx.Response:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=follow_redirects) as c:
        return await c.get(url, params=params, headers={**BROWSER_HEADERS, **(headers or {})})


def _rospatent_headers(token: str | None) -> dict[str, str]:
    tok = (token or ROSPATENT_TOKEN).strip()
    if not tok:
        raise SourceError(
            "Не задан токен Поисковой платформы Роспатента. Укажите переменную окружения "
            "ROSPATENT_TOKEN на хостинге или передайте параметр token. Получить токен: "
            "searchplatform.rospatent.gov.ru → личный кабинет → раздел API."
        )
    if tok.lower().startswith("bearer "):
        tok = tok[7:]
    return {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Источник: Поисковая платформа Роспатента (официальный API, нужен токен)
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_rospatent_hit(hit: dict) -> dict[str, Any]:
    biblio_ru = _dig(hit, "biblio", "ru", default={}) or {}
    biblio_en = _dig(hit, "biblio", "en", default={}) or {}
    common = hit.get("common", {}) or {}
    snippet = hit.get("snippet", {}) or {}
    doc_id = hit.get("id", "")

    number = ""
    if common.get("document_number"):
        number = f"{common.get('publishing_office', 'RU')}{common['document_number']}{common.get('kind', '')}"
    elif doc_id:
        number = doc_id.split("_")[0]

    return {
        "id": doc_id,
        "number": number,
        "title": _clean(biblio_ru.get("title") or snippet.get("title") or biblio_en.get("title")),
        "abstract": _clean(snippet.get("description") or _dig(hit, "abstract", "ru")),
        "patentees": _names(biblio_ru.get("patentee") or biblio_en.get("patentee")),
        "applicants": _names(biblio_ru.get("applicant") or biblio_en.get("applicant")),
        "inventors": _names(biblio_ru.get("inventor") or biblio_en.get("inventor")),
        "publication_date": common.get("publication_date", ""),
        "application_date": _dig(common, "application", "filing_date", default=""),
        "ipc": [_dig(i, "fullname", default="") for i in (_dig(common, "classification", "ipc", default=[]) or [])],
        "url": f"{ROSPATENT_WEB}/docs/{doc_id}" if doc_id else "",
    }


async def _rospatent_search_raw(
    *, query: str, natural_language: bool, limit: int, offset: int,
    date_from: str | None, date_to: str | None, datasets: list[str] | None,
    token: str | None,
) -> dict[str, Any]:
    headers = _rospatent_headers(token)
    payload: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "pre_tag": "",
        "post_tag": "",
    }
    payload["qn" if natural_language else "q"] = query
    if datasets:
        payload["datasets"] = datasets
    if date_from or date_to:
        rng: dict[str, str] = {}
        if date_from:
            rng["gte"] = date_from.replace("-", "")
        if date_to:
            rng["lte"] = date_to.replace("-", "")
        payload["filter"] = {"date_published": {"range": rng}}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        r = await c.post(f"{ROSPATENT_API}/search", json=payload, headers=headers)

    if r.status_code in (401, 403):
        raise SourceError(f"Роспатент отклонил токен (HTTP {r.status_code}). Проверьте ROSPATENT_TOKEN.")
    if r.status_code >= 400:
        raise SourceError(f"Роспатент вернул HTTP {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except json.JSONDecodeError as e:
        raise SourceError(f"Роспатент вернул не-JSON ответ: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# Источник: Google Patents (неофициальный XHR, без токена) — запасной вариант
# ──────────────────────────────────────────────────────────────────────────────


async def _google_patents_search(query: str, limit: int, assignee: str | None = None) -> list[dict]:
    inner: dict[str, Any] = {"country": "RU", "num": max(10, min(limit, 100))}
    if query:
        inner["q"] = query
    if assignee:
        inner["assignee"] = assignee
    params = {"url": urlencode(inner), "exp": ""}
    r = await _http_get(GOOGLE_PATENTS_XHR, params=params, headers={"Accept": "application/json"})
    if r.status_code != 200:
        raise SourceError(f"Google Patents вернул HTTP {r.status_code} (возможна блокировка IP облака)")
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise SourceError("Google Patents вернул не-JSON (вероятно, капча)") from e

    out = []
    for cluster in _dig(data, "results", "cluster", default=[]) or []:
        for item in cluster.get("result", []) or []:
            p = item.get("patent", {}) or {}
            pub = p.get("publication_number", "")
            out.append({
                "id": pub,
                "number": pub,
                "title": _clean(p.get("title")),
                "abstract": _clean(p.get("snippet")),
                "patentees": [_clean(p.get("assignee"))] if p.get("assignee") else [],
                "inventors": [_clean(p.get("inventor"))] if p.get("inventor") else [],
                "publication_date": p.get("publication_date", ""),
                "priority_date": p.get("priority_date", ""),
                "url": f"https://patents.google.com/patent/{pub}/ru" if pub else "",
            })
    return out[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# Источник: Яндекс.Патенты (парсинг HTML)
# ──────────────────────────────────────────────────────────────────────────────

_YANDEX_DOC_RE = re.compile(r"/patents/doc/((?:RU|SU)\d+[A-Z]\d?)(?:_(\d{8}))?")


async def _yandex_patents_search(query: str, limit: int) -> list[dict]:
    r = await _http_get(YANDEX_PATENTS_URL, params={"text": query})
    final_url = str(r.url)
    if "showcaptcha" in final_url or "captcha" in r.text[:5000].lower():
        raise SourceError("Яндекс показал капчу (запросы с облачных IP часто блокируются)")
    if r.status_code != 200:
        raise SourceError(f"Яндекс.Патенты вернул HTTP {r.status_code}")

    soup = BeautifulSoup(r.text, "html.parser")
    results: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        m = _YANDEX_DOC_RE.search(a["href"])
        if not m:
            continue
        number = m.group(1)
        title = _clean(a.get_text(" "))
        entry = results.setdefault(number, {
            "id": number, "number": number, "title": "", "abstract": "",
            "publication_date": m.group(2) or "",
            "url": f"https://yandex.ru/patents/doc/{number}" + (f"_{m.group(2)}" if m.group(2) else ""),
        })
        if title and len(title) > len(entry["title"]):
            entry["title"] = title
        # Реферат: ближайший крупный текстовый блок в карточке выдачи
        card = a.find_parent(["li", "article", "div"])
        if card and not entry["abstract"]:
            texts = [t for t in (_clean(x) for x in card.stripped_strings) if len(t) > 120]
            if texts:
                entry["abstract"] = texts[0][:1500]
        if len(results) >= limit:
            break

    if not results:
        raise SourceError(
            "Яндекс.Патенты не вернул результатов. С лета 2025 сервис приостановлен "
            "(открывается страница-заглушка), поэтому используются запасные источники."
        )
    return list(results.values())


# ──────────────────────────────────────────────────────────────────────────────
# Источник: открытые реестры ФИПС (HTML-карточка документа по номеру, без токена)
# ──────────────────────────────────────────────────────────────────────────────

FIPS_DB = {
    "invention": "RUPAT",       # патенты на изобретения
    "utility_model": "RUPM",    # патенты на полезные модели
    "application": "RUPATAP",   # заявки на изобретения
    "design": "RUDE",           # промышленные образцы
    "program": "EVM",           # программы для ЭВМ
    "database": "DB",           # базы данных
}


def _doc_digits(number: str) -> str:
    """'RU2804073C1' / 'RU 2 804 073' / '2804073' → '2804073' (без кода вида C1/U1)."""
    n = re.sub(r"[\s\u00a0]", "", number or "").upper()
    n = re.sub(r"^(RU|SU)", "", n)
    m = re.match(r"\d+", n)
    return m.group(0) if m else ""


def _fips_field(text: str, code: str) -> str:
    """Извлекает значение поля с INID-кодом, напр. (73) Патентообладатель(и): ..."""
    m = re.search(rf"\({code}\)\s*([^:]{{0,80}}:)?\s*(.+?)(?=\(\d{{2}}\)|Реферат|Формула|$)", text, re.S)
    return _clean(m.group(2)) if m else ""


async def _fips_fetch(number: str, kind: str) -> dict[str, Any]:
    db = FIPS_DB.get(kind, kind)
    digits = _doc_digits(number)
    if not digits:
        raise SourceError("Номер документа должен содержать цифры, например 2754321 или RU2754321C1")
    params = {"DB": db, "DocNumber": digits, "TypeFile": "html"}
    url = f"{FIPS_DOC_URL}?{urlencode(params)}"
    r = await _http_get(FIPS_DOC_URL, params=params)
    if r.status_code != 200:
        raise SourceError(f"ФИПС вернул HTTP {r.status_code}")

    raw = r.content
    html = None
    for enc in (r.encoding, "windows-1251", "utf-8"):
        if not enc:
            continue
        try:
            html = raw.decode(enc)
            if "Документ с данным номером не найден" in html or "<html" in html.lower():
                break
        except (LookupError, UnicodeDecodeError):
            continue
    html = html or raw.decode("utf-8", errors="replace")

    if "не найден" in html.lower() and "документ" in html.lower() and len(html) < 4000:
        raise SourceError(f"В реестре {db} документ {digits} не найден")

    soup = BeautifulSoup(html, "html.parser")
    text = _clean(soup.get_text(" "))

    def by_id(*ids: str) -> str:
        for i in ids:
            el = soup.find(id=i)
            if el:
                t = _clean(el.get_text(" "))
                if t:
                    return t
        return ""

    title = by_id("B542") or _fips_field(text, "54")
    abstract = by_id("Abs")
    if not abstract:
        m = re.search(r"Реферат\s*:?\s*(.+?)(?=Формула изобретения|Формула полезной модели|Описание|$)", text, re.S)
        abstract = _clean(m.group(1))[:3000] if m else ""
    claims = ""
    m = re.search(r"(Формула (?:изобретения|полезной модели).+?)(?=Описание изобретения|Описание полезной модели|$)", text, re.S)
    if m:
        claims = _clean(m.group(1))[:6000]

    status = by_id("StatusR", "Status")
    if not status:
        m = re.search(r"Статус:\s*(.+?)(?:\(|Пошлина|$)", text)
        status = _clean(m.group(1))[:200] if m else ""

    return {
        "ok": True,
        "source": "ФИПС, открытые реестры",
        "registry": db,
        "number": digits,
        "title": title,
        "status": status,
        "patentees": _fips_field(text, "73"),
        "authors": _fips_field(text, "72"),
        "applicants": _fips_field(text, "71"),
        "application": _fips_field(text, "21") or _fips_field(text, "21\\)\\(22"),
        "ipc": _fips_field(text, "51"),
        "abstract": abstract,
        "claims": claims,
        "url": url,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Регламенты
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "obzor": ("обзор", "общее", "overview", "всё", "все", "all"),
    "formula": ("формула", "claims", "пункт", "независим"),
    "opisanie": ("описание", "description", "уровень техники", "раскрытие"),
    "referat": ("реферат", "abstract"),
    "chertezhi": ("чертеж", "чертёж", "рисун", "drawing"),
    "posledovatelnosti": ("последовательн", "st.26", "sequence", "нуклеотид", "аминокислот"),
    "poshliny": ("пошлин", "fee", "оплат", "стоимость", "льгот", "скидк"),
    "podacha": ("подач", "процесс", "этап", "экспертиз", "filing", "сроки", "срок"),
    "msp": ("мсп", "малый", "средний", "sme", "предпринимат", "стартап"),
    "poleznaya_model": ("полезн", "utility"),
    "sluzhebnye": ("служебн", "работодател", "вуз", "университет", "кфу"),
    "istochniki": ("источник", "норматив", "приказ", "закон", "гк", "sources"),
}


def _load_regulations() -> dict[str, dict[str, str]]:
    """Читает regulations.md: разделы вида  '## key | Заголовок'."""
    try:
        content = REGULATIONS_FILE.read_text(encoding="utf-8")
    except OSError as e:
        raise SourceError(f"Не удалось прочитать файл регламентов {REGULATIONS_FILE}: {e}") from e

    sections: dict[str, dict[str, str]] = {}
    meta = ""
    current_key, current_title, buf = None, "", []
    for line in content.splitlines():
        if line.startswith("## "):
            if current_key:
                sections[current_key] = {"title": current_title, "text": "\n".join(buf).strip()}
            header = line[3:].strip()
            key, _, title = header.partition("|")
            current_key, current_title, buf = key.strip(), title.strip() or key.strip(), []
        elif current_key is None:
            meta += line + "\n"
        else:
            buf.append(line)
    if current_key:
        sections[current_key] = {"title": current_title, "text": "\n".join(buf).strip()}
    sections["_meta"] = {"title": "Сведения о файле", "text": meta.strip()}
    return sections


# ──────────────────────────────────────────────────────────────────────────────
# MCP-сервер и инструменты
# ──────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="patents-ru",
    instructions=(
        "Инструменты для патентного поиска по РФ (Роспатент/ФИПС, Яндекс.Патенты с запасными "
        "источниками, патенты КФУ) и справка по регламентам подачи заявок в Роспатент. "
        "Для поиска сначала используй rospatent_search (самый полный), карточку по номеру — "
        "fips_get_document. Перед советами по оформлению заявки вызывай get_patenting_guidelines. "
        "Суммы пошлин всегда сверяй с калькулятором Роспатента и сообщай пользователю дату редакции."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path=MCP_PATH,
    # Сервер стоит за HTTPS-прокси хостинга с произвольным доменом → проверка Host отключена.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
async def search_yandex_patents(query: str, limit: int = 10) -> dict[str, Any]:
    """Поиск российских патентов по ключевым словам (как в Яндекс.Патентах).

    Пытается получить выдачу Яндекс.Патентов; если сервис недоступен (с 2025 г. он
    приостановлен) или показывает капчу — автоматически переключается на Поисковую
    платформу Роспатента (если задан токен), затем на Google Patents (страна RU).

    Args:
        query: поисковый запрос на русском, напр. "моноклональное антитело NaPi2b".
        limit: сколько результатов вернуть (1–50).
    Returns:
        JSON: source (какой источник сработал), warnings, results[] с полями
        number, title, abstract, patentees, publication_date, url.
    """
    limit = _clamp(limit, 1, 50)
    if not query or not query.strip():
        return _error("input", "Пустой запрос")
    warnings: list[str] = []

    try:
        res = await _yandex_patents_search(query, limit)
        return {"ok": True, "source": "Яндекс.Патенты", "warnings": warnings, "count": len(res), "results": res}
    except (SourceError, httpx.HTTPError) as e:
        warnings.append(f"Яндекс.Патенты: {e}")
    except Exception as e:  # noqa: BLE001
        log.exception("yandex")
        warnings.append(f"Яндекс.Патенты: непредвиденная ошибка {type(e).__name__}")

    if ROSPATENT_TOKEN:
        try:
            data = await _rospatent_search_raw(
                query=query, natural_language=True, limit=limit, offset=0,
                date_from=None, date_to=None, datasets=["ru_till_1994", "ru_since_1994"], token=None,
            )
            res = [_normalize_rospatent_hit(h) for h in data.get("hits", [])]
            return {"ok": True, "source": "Поисковая платформа Роспатента", "warnings": warnings,
                    "total": data.get("total"), "count": len(res), "results": res}
        except (SourceError, httpx.HTTPError) as e:
            warnings.append(f"Роспатент: {e}")
        except Exception as e:  # noqa: BLE001
            log.exception("rospatent fallback")
            warnings.append(f"Роспатент: непредвиденная ошибка {type(e).__name__}")
    else:
        warnings.append("Роспатент пропущен: не задан ROSPATENT_TOKEN")

    try:
        res = await _google_patents_search(query, limit)
        return {"ok": True, "source": "Google Patents (RU)", "warnings": warnings, "count": len(res), "results": res}
    except (SourceError, httpx.HTTPError) as e:
        warnings.append(f"Google Patents: {e}")
    except Exception as e:  # noqa: BLE001
        log.exception("google")
        warnings.append(f"Google Patents: непредвиденная ошибка {type(e).__name__}")

    return _error("all", "Ни один источник не ответил", warnings=warnings,
                  manual_search_url=f"{ROSPATENT_WEB}/?q={quote(query)}")


@mcp.tool()
async def rospatent_search(
    query: str,
    limit: int = 10,
    offset: int = 0,
    natural_language: bool = True,
    date_from: str = "",
    date_to: str = "",
    only_russian: bool = True,
    token: str = "",
) -> dict[str, Any]:
    """Поиск по Поисковой платформе Роспатента (официальный API ФИПС/Роспатента).

    Args:
        query: запрос. При natural_language=True — обычный текст (семантический поиск);
               при False — язык запросов платформы (AND/OR/NOT, кавычки, поля).
        limit: число результатов (1–100). offset: смещение для постраничного вывода.
        date_from, date_to: границы даты публикации в формате YYYY-MM-DD (необязательно).
        only_russian: True — искать только в российских массивах (RU до/после 1994).
        token: JWT-токен API; если пусто — берётся из переменной окружения ROSPATENT_TOKEN.
    Returns:
        JSON: total, results[] (id, number, title, abstract, patentees, inventors, ipc, url).
    """
    try:
        data = await _rospatent_search_raw(
            query=query, natural_language=natural_language,
            limit=_clamp(limit, 1, 100), offset=_clamp(offset, 0, 10_000),
            date_from=date_from or None, date_to=date_to or None,
            datasets=["ru_till_1994", "ru_since_1994"] if only_russian else None,
            token=token or None,
        )
        hits = [_normalize_rospatent_hit(h) for h in data.get("hits", [])]
        return {"ok": True, "source": "Поисковая платформа Роспатента",
                "total": data.get("total"), "count": len(hits), "results": hits}
    except (SourceError, httpx.HTTPError) as e:
        return _error("rospatent", str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("rospatent_search")
        return _error("rospatent", f"Непредвиденная ошибка: {type(e).__name__}: {e}")


@mcp.tool()
async def rospatent_get_document(doc_id: str, include_full_text: bool = False, token: str = "") -> dict[str, Any]:
    """Полный документ из Поисковой платформы Роспатента по ID.

    Args:
        doc_id: ID из результатов rospatent_search, напр. "RU2754321C1_20210901".
        include_full_text: True — вернуть также формулу и описание (может быть длинно).
        token: JWT-токен; по умолчанию ROSPATENT_TOKEN.
    """
    try:
        headers = _rospatent_headers(token or None)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
            r = await c.get(f"{ROSPATENT_API}/docs/{quote(doc_id.strip())}", headers=headers)
        if r.status_code == 404:
            return _error("rospatent", f"Документ {doc_id} не найден")
        if r.status_code >= 400:
            return _error("rospatent", f"HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        result = _normalize_rospatent_hit(d)
        result["abstract"] = _clean(_dig(d, "abstract", "ru") or _dig(d, "abstract", "en")) or result["abstract"]
        if include_full_text:
            result["claims"] = _clean(_dig(d, "claims", "ru") or _dig(d, "claims", "en"))[:20000]
            result["description"] = _clean(_dig(d, "description", "ru") or _dig(d, "description", "en"))[:30000]
        return {"ok": True, "source": "Поисковая платформа Роспатента", "document": result}
    except (SourceError, httpx.HTTPError) as e:
        return _error("rospatent", str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("rospatent_get_document")
        return _error("rospatent", f"Непредвиденная ошибка: {type(e).__name__}: {e}")


@mcp.tool()
async def fips_get_document(number: str, kind: str = "invention") -> dict[str, Any]:
    """Карточка документа из открытых реестров ФИПС по номеру (без токена).

    Возвращает название, статус (действует/прекратил действие), правообладателей,
    авторов, МПК, реферат и формулу.

    Args:
        number: номер патента/заявки, напр. "2754321" или "RU2754321C1".
        kind: тип реестра — invention (изобретения), utility_model (полезные модели),
              application (заявки на изобретения), design, program, database.
    """
    try:
        return await _fips_fetch(number, kind)
    except (SourceError, httpx.HTTPError) as e:
        fips_error = str(e) or type(e).__name__
    except Exception as e:  # noqa: BLE001
        log.exception("fips_get_document")
        fips_error = f"Непредвиденная ошибка: {type(e).__name__}"

    # Запасной путь: ФИПС часто блокирует зарубежные IP — берём документ из API Роспатента
    if not ROSPATENT_TOKEN:
        return _error("fips", fips_error)
    try:
        doc = await _rospatent_find_by_number(number)
        doc["warnings"] = [f"ФИПС недоступен с сервера ({fips_error}); данные из Поисковой платформы Роспатента. "
                           "Статус действия патента смотрите на fips.ru."]
        return doc
    except (SourceError, httpx.HTTPError) as e2:
        return _error("fips", f"ФИПС: {fips_error}; Роспатент: {e2}")
    except Exception as e2:  # noqa: BLE001
        log.exception("fips fallback")
        return _error("fips", f"ФИПС: {fips_error}; Роспатент: {type(e2).__name__}")


async def _rospatent_find_by_number(number: str) -> dict[str, Any]:
    digits = _doc_digits(number)
    if not digits:
        raise SourceError("Номер должен содержать цифры")
    headers = _rospatent_headers(None)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        hits: list[dict] = []
        for q in (f"PN=RU{digits}", f"DN={digits}", f"RU{digits}", digits):
            r = await c.post(f"{ROSPATENT_API}/search", headers=headers,
                             json={"q": q, "limit": 20, "datasets": ["ru_till_1994", "ru_since_1994"]})
            if r.status_code in (401, 403):
                raise SourceError(f"токен отклонён (HTTP {r.status_code})")
            if r.status_code >= 400:
                continue  # вариант синтаксиса не поддержан — пробуем следующий
            hits = [h for h in r.json().get("hits", [])
                    if re.match(rf"(RU|SU){digits}[A-Z]", h.get("id", ""))]
            if hits:
                break
        if not hits:
            raise SourceError(f"документ {digits} не найден")
        hits.sort(key=lambda h: h.get("id", ""), reverse=True)  # самая поздняя публикация
        doc_id = hits[0]["id"]
        r = await c.get(f"{ROSPATENT_API}/docs/{quote(doc_id)}", headers=headers)
        d = r.json() if r.status_code == 200 else hits[0]
    res = _normalize_rospatent_hit(d)
    res["abstract"] = _clean(_dig(d, "abstract", "ru")) or res["abstract"]
    res["claims"] = _clean(_dig(d, "claims", "ru"))[:6000]
    res["other_publications"] = [h.get("id") for h in hits[1:]]
    return {"ok": True, "source": "Поисковая платформа Роспатента (вместо ФИПС)", **res}


@mcp.tool()
async def search_kfu_patents(topic: str = "", limit: int = 20, date_from: str = "") -> dict[str, Any]:
    """Патенты, где правообладатель — Казанский (Приволжский) федеральный университет (КФУ).

    Ищет через Поисковую платформу Роспатента (если задан токен) с фильтрацией по
    правообладателю; иначе — через Google Patents по полю assignee.

    Args:
        topic: тема для сужения, напр. "антитела", "рак яичника", "ПЦР". Пусто — все патенты КФУ.
        limit: сколько результатов вернуть (1–50).
        date_from: публикации не раньше этой даты, YYYY-MM-DD (необязательно).
    """
    limit = _clamp(limit, 1, 50)
    warnings: list[str] = []

    def is_kfu(names: list[str]) -> bool:
        return any(all(m in n.lower() for m in KFU_NAME_MARKERS) for n in names)

    if ROSPATENT_TOKEN:
        try:
            query = f"{KFU_DEFAULT_QUERY} {topic}".strip()
            collected: list[dict] = []
            for page in range(3):  # до 300 кандидатов, фильтруем по правообладателю
                data = await _rospatent_search_raw(
                    query=query, natural_language=True, limit=100, offset=page * 100,
                    date_from=date_from or None, date_to=None,
                    datasets=["ru_till_1994", "ru_since_1994"], token=None,
                )
                hits = [_normalize_rospatent_hit(h) for h in data.get("hits", [])]
                collected += [h for h in hits if is_kfu(h["patentees"] + h["applicants"])]
                if len(collected) >= limit or len(hits) < 100:
                    break
            if collected:
                return {"ok": True, "source": "Поисковая платформа Роспатента (фильтр: правообладатель КФУ)",
                        "warnings": warnings, "count": len(collected[:limit]), "results": collected[:limit]}
            warnings.append("Роспатент: среди найденного нет документов с правообладателем КФУ")
        except (SourceError, httpx.HTTPError) as e:
            warnings.append(f"Роспатент: {e}")
        except Exception as e:  # noqa: BLE001
            log.exception("kfu rospatent")
            warnings.append(f"Роспатент: непредвиденная ошибка {type(e).__name__}")
    else:
        warnings.append("Роспатент пропущен: не задан ROSPATENT_TOKEN")

    try:
        res = await _google_patents_search(topic, limit * 2, assignee="Казанский федеральный университет")
        filtered = [r for r in res if is_kfu(r["patentees"])] or res
        if date_from:
            df = date_from.replace("-", "")
            filtered = [r for r in filtered if (r.get("publication_date") or "").replace("-", "") >= df]
        return {"ok": True, "source": "Google Patents (assignee: КФУ)", "warnings": warnings,
                "count": len(filtered[:limit]), "results": filtered[:limit]}
    except (SourceError, httpx.HTTPError) as e:
        warnings.append(f"Google Patents: {e}")
    except Exception as e:  # noqa: BLE001
        log.exception("kfu google")
        warnings.append(f"Google Patents: непредвиденная ошибка {type(e).__name__}")

    return _error("all", "Не удалось получить патенты КФУ ни из одного источника", warnings=warnings,
                  manual_search_url=f"{ROSPATENT_WEB}/?q={quote(KFU_DEFAULT_QUERY + ' ' + topic)}")


@mcp.tool()
async def get_patenting_guidelines(topic: str = "обзор") -> dict[str, Any]:
    """Регламенты Роспатента для подготовки и подачи заявки (РФ).

    Темы: обзор, формула, описание, реферат, чертежи, последовательности (ST.26),
    пошлины, подача (этапы и сроки), мсп, полезная модель, служебные изобретения,
    источники. Значение "все" вернёт весь справочник. Можно передать несколько тем
    через запятую: "формула, реферат".
    """
    try:
        sections = _load_regulations()
    except SourceError as e:
        return _error("regulations", str(e))

    meta = sections.pop("_meta", {}).get("text", "")
    wanted = [t.strip().lower() for t in (topic or "обзор").split(",") if t.strip()]

    if any(w in ("все", "всё", "all") for w in wanted):
        keys = list(sections)
    else:
        keys = []
        for w in wanted:
            for key, aliases in TOPIC_ALIASES.items():
                if key in sections and (w == key or any(a in w or w in a for a in aliases)):
                    if key not in keys:
                        keys.append(key)
            for key, sec in sections.items():  # поиск по заголовку раздела
                if w in sec["title"].lower() and key not in keys:
                    keys.append(key)

    if not keys:
        return {"ok": False, "source": "regulations",
                "error": f"Тема «{topic}» не найдена",
                "available_topics": {k: v["title"] for k, v in sections.items()}}

    return {
        "ok": True,
        "source": f"Локальный справочник ({REGULATIONS_FILE.name})",
        "note": meta,
        "sections": [{"key": k, "title": sections[k]["title"], "text": sections[k]["text"]} for k in keys],
        "available_topics": {k: v["title"] for k, v in sections.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI-приложение
# ──────────────────────────────────────────────────────────────────────────────

mcp_app = mcp.streamable_http_app()  # создаёт session_manager — вызывать до lifespan


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp.session_manager.run():
        log.info("MCP endpoint: %s | Rospatent token: %s", MCP_PATH, "задан" if ROSPATENT_TOKEN else "НЕ задан")
        yield


app = FastAPI(title="Patent MCP Server", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/")
async def root() -> JSONResponse:
    # Намеренно не раскрываем путь MCP-эндпоинта
    return JSONResponse({"service": "patent-mcp", "status": "ok"})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "rospatent_token": bool(ROSPATENT_TOKEN),
                         "regulations_file": REGULATIONS_FILE.exists()})


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return JSONResponse({"error": "internal_error", "detail": type(exc).__name__}, status_code=500)


# MCP монтируется последним, чтобы маршруты выше имели приоритет
app.mount("/", mcp_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

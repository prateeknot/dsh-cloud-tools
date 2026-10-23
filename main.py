import os, io, json, re, csv, httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
from contextlib import redirect_stdout, redirect_stderr

app = FastAPI(title="DSH Cloud Tools", version="2.2.0")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")


def get_db():
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None


@app.on_event("startup")
def startup():
    conn = get_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS data_lake (id SERIAL PRIMARY KEY, source TEXT, content TEXT, created_at TIMESTAMP DEFAULT NOW())")
            conn.commit(); cur.close(); conn.close()
            print("DB ready")
        except Exception as e:
            print(f"DB init failed: {e}"); conn.close()


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

class StoreRequest(BaseModel):
    source: str
    content: str

class RunCodeRequest(BaseModel):
    code: str

class TextProcessRequest(BaseModel):
    text: str
    operation: str = "summary"

class WeatherRequest(BaseModel):
    city: str

class WikipediaRequest(BaseModel):
    topic: str
    sentences: int = 3

class NewsRequest(BaseModel):
    topic: str = "technology"
    max_results: int = 5

class GithubRepoRequest(BaseModel):
    owner: str
    repo: str

class JsonFormatRequest(BaseModel):
    data: str

class RegexRequest(BaseModel):
    pattern: str
    text: str

class MarkdownRequest(BaseModel):
    text: str

class CsvToJsonRequest(BaseModel):
    csv_data: str

class UrlShortenRequest(BaseModel):
    url: str

class SentimentRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    source: str = "auto"
    target: str = "en"

class ArxivRequest(BaseModel):
    query: str
    max_results: int = 5


@app.get("/")
async def root():
    return {"status": "online", "version": "2.2.0", "tools": [
        "web_search", "fetch_url", "run_python", "text_process", "store_data",
        "query_data", "health", "wikipedia", "news", "weather", "arxiv",
        "github_repo", "code_explain", "json_format", "regex_test",
        "pdf_extract", "ocr_image", "markdown_to_html", "csv_to_json",
        "url_shorten", "sentiment", "translate"]}


@app.get("/health")
async def health():
    db_ok = False
    conn = get_db()
    if conn:
        try:
            cur = conn.cursor(); cur.execute("SELECT 1"); cur.close(); conn.close()
            db_ok = True
        except Exception:
            pass
    return {"api": True, "database": db_ok}


@app.post("/search")
async def web_search(req: SearchRequest):
    results, debug = [], []
    hdr = {"User-Agent": UA, "Accept": "application/json"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        # 1) Wikipedia
        try:
            r = await client.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": req.query,
                "srlimit": min(req.max_results, 5), "format": "json", "utf8": 1
            }, headers=hdr)
            e = {"src": "wikipedia", "status": r.status_code}
            if r.status_code == 200:
                hits = r.json().get("query", {}).get("search", [])
                e["hits"] = len(hits)
                for it in hits:
                    t = it.get("title", "")
                    results.append({
                        "title": t,
                        "url": f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}",
                        "snippet": re.sub(r"<[^>]+>", "", it.get("snippet", ""))[:300],
                        "source": "wikipedia"})
            debug.append(e)
        except Exception as ex:
            debug.append({"src": "wikipedia", "error": f"{type(ex).__name__}: {ex}"})

        # 2) DDG Instant Answer
        try:
            r = await client.get("https://api.duckduckgo.com/",
                                 params={"q": req.query, "format": "json", "no_redirect": 1},
                                 headers=hdr)
            e = {"src": "duckduckgo", "status": r.status_code}
            if r.status_code == 200:
                d = r.json()
                e["abstract"] = bool(d.get("AbstractText"))
                e["related"] = len(d.get("RelatedTopics", []))
                if d.get("AbstractText") and d.get("AbstractURL"):
                    results.append({"title": d.get("Heading", req.query),
                                    "url": d["AbstractURL"],
                                    "snippet": d["AbstractText"][:300],
                                    "source": "duckduckgo"})
                for t in d.get("RelatedTopics", [])[:req.max_results]:
                    if t.get("Text") and t.get("FirstURL"):
                        results.append({"title": t["Text"][:80], "url": t["FirstURL"],
                                        "snippet": t["Text"][:300], "source": "duckduckgo"})
                for x in d.get("Results", [])[:3]:
                    if x.get("Text") and x.get("FirstURL"):
                        results.append({"title": x["Text"][:80], "url": x["FirstURL"],
                                        "snippet": x["Text"][:300], "source": "duckduckgo"})
            debug.append(e)
        except Exception as ex:
            debug.append({"src": "duckduckgo", "error": f"{type(ex).__name__}: {ex}"})

        # 3) DDG HTML fallback
        if len(results) < req.max_results:
            try:
                r = await client.get("https://html.duckduckgo.com/html/",
                                     params={"q": req.query},
                                     headers={"User-Agent": UA,
                                              "Accept-Language": "en-US,en;q=0.9"})
                e = {"src": "ddg_html", "status": r.status_code, "bytes": len(r.text)}
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    blocks = soup.select(".result")
                    e["blocks"] = len(blocks)
                    for b in blocks:
                        a = b.select_one(".result__a") or b.select_one(".result__title a")
                        sn = b.select_one(".result__snippet")
                        if not a:
                            continue
                        href = a.get("href", "")
                        if "uddg=" in href:
                            from urllib.parse import unquote, parse_qs, urlparse
                            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
                        results.append({"title": a.get_text(strip=True), "url": href,
                                        "snippet": sn.get_text(strip=True) if sn else "",
                                        "source": "ddg_html"})
                        if len(results) >= req.max_results:
                            break
                debug.append(e)
            except Exception as ex:
                debug.append({"src": "ddg_html", "error": f"{type(ex).__name__}: {ex}"})

    seen, unique = set(), []
    for x in results:
        if x["url"] and x["url"] not in seen:
            seen.add(x["url"]); unique.append(x)
    return {"query": req.query, "results": unique[:req.max_results],
            "total_found": len(unique), "debug": debug}


@app.post("/fetch")
async def fetch_url(url: str):
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        r = await client.get(url, headers={"User-Agent": UA})
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        return {"url": url, "status": r.status_code,
                "content": soup.get_text(separator="\n", strip=True)[:10000]}


@app.post("/run")
async def run_python(req: RunCodeRequest):
    out, err, res = io.StringIO(), None, None
    try:
        with redirect_stdout(out), redirect_stderr(out):
            ns = {}
            exec(req.code, {"__builtins__": __builtins__}, ns)
            if "result" in ns:
                res = str(ns["result"])
    except Exception as ex:
        err = str(ex)
    return {"output": out.getvalue(), "error": err, "result": res}


@app.post("/text")
async def text_process(req: TextProcessRequest):
    t, op = req.text, req.operation
    if op == "summary":
        w = t.split()
        return {"word_count": len(w), "summary": " ".join(w[:100]) + ("..." if len(w) > 100 else "")}
    if op == "words":
        return {"word_count": len(t.split()), "char_count": len(t)}
    if op == "uppercase":
        return {"result": t.upper()}
    if op == "lowercase":
        return {"result": t.lower()}
    if op == "reverse":
        return {"result": t[::-1]}
    raise HTTPException(400, f"Unknown operation: {op}")


@app.post("/store")
async def store_data(req: StoreRequest):
    conn = get_db()
    if not conn:
        raise HTTPException(503, "Database not connected")
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO data_lake (source, content) VALUES (%s, %s) RETURNING id",
                    (req.source, req.content))
        rid = cur.fetchone()["id"]
        conn.commit(); cur.close(); conn.close()
        return {"stored": True, "id": rid}
    except Exception as ex:
        conn.close(); raise HTTPException(500, str(ex))


@app.get("/query")
async def query_data(q: str, limit: int = 10):
    conn = get_db()
    if not conn:
        raise HTTPException(503, "Database not connected")
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, source, content, created_at FROM data_lake "
                    "WHERE content ILIKE %s OR source ILIKE %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (f"%{q}%", f"%{q}%", limit))
        rows = cur.fetchall(); cur.close(); conn.close()
        return {"results": [dict(r) for r in rows], "count": len(rows)}
    except Exception as ex:
        conn.close(); raise HTTPException(500, str(ex))


@app.post("/wikipedia")
async def wikipedia_search(req: WikipediaRequest):
    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": UA}) as c:
            r = await c.get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                            + req.topic.replace(" ", "_"))
            if r.status_code != 200:
                return {"topic": req.topic, "error": f"HTTP {r.status_code}", "summary": ""}
            d = r.json()
            return {"topic": req.topic, "title": d.get("title"),
                    "summary": d.get("extract", ""),
                    "url": d.get("content_urls", {}).get("desktop", {}).get("page", "")}
    except Exception as ex:
        return {"topic": req.topic, "error": str(ex), "summary": ""}


@app.post("/news")
async def news_headlines(req: NewsRequest):
    feeds = {"technology": "https://feeds.arstechnica.com/arstechnica/technology-lab",
             "science": "https://feeds.arstechnica.com/arstechnica/science",
             "world": "http://feeds.bbci.co.uk/news/world/rss.xml",
             "business": "http://feeds.bbci.co.uk/news/business/rss.xml",
             "general": "http://feeds.bbci.co.uk/news/rss.xml"}
    try:
        import feedparser
        p = feedparser.parse(feeds.get(req.topic.lower(), feeds["general"]))
        arts = [{"title": e.get("title", ""), "link": e.get("link", ""),
                 "published": e.get("published", ""),
                 "summary": e.get("summary", "")[:300]} for e in p.entries[:req.max_results]]
        return {"topic": req.topic, "articles": arts}
    except Exception as ex:
        return {"error": str(ex), "articles": []}


@app.post("/weather")
async def weather(req: WeatherRequest):
    async with httpx.AsyncClient(timeout=15) as c:
        g = await c.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": req.city, "count": 1})
        gd = g.json()
        if not gd.get("results"):
            return {"error": f"City '{req.city}' not found"}
        loc = gd["results"][0]
        w = await c.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": loc["latitude"], "longitude": loc["longitude"],
            "current_weather": True, "timezone": "auto"})
        cw = w.json().get("current_weather", {})
        return {"city": loc["name"], "country": loc.get("country", ""),
                "temperature_c": cw.get("temperature"), "windspeed_kmh": cw.get("windspeed"),
                "weathercode": cw.get("weathercode"), "time": cw.get("time")}


@app.post("/arxiv")
async def arxiv_search(req: ArxivRequest):
    async with httpx.AsyncClient(timeout=25, headers={"User-Agent": UA}) as c:
        r = await c.get("http://export.arxiv.org/api/query",
                        params={"search_query": f"all:{req.query}",
                                "max_results": req.max_results})
        soup = BeautifulSoup(r.text, "xml")
        papers = []
        for e in soup.find_all("entry"):
            papers.append({
                "title": (e.find("title").text.strip().replace("\n", " ") if e.find("title") else ""),
                "authors": [a.find("name").text for a in e.find_all("author") if a.find("name")][:3],
                "summary": (e.find("summary").text.strip().replace("\n", " ")[:300] if e.find("summary") else ""),
                "url": (e.find("id").text.strip() if e.find("id") else "")})
        return {"query": req.query, "papers": papers}


@app.post("/github_repo")
async def github_repo(req: GithubRepoRequest):
    async with httpx.AsyncClient(timeout=15, headers={
            "Accept": "application/vnd.github.v3+json", "User-Agent": UA}) as c:
        r = await c.get(f"https://api.github.com/repos/{req.owner}/{req.repo}")
        if r.status_code != 200:
            return {"error": f"Repo not found or rate limited ({r.status_code})"}
        d = r.json()
        return {"name": d.get("full_name"), "description": d.get("description"),
                "stars": d.get("stargazers_count"), "forks": d.get("forks_count"),
                "language": d.get("language"),
                "license": (d.get("license") or {}).get("spdx_id"),
                "url": d.get("html_url"), "topics": d.get("topics", [])[:10]}


@app.post("/code_explain")
async def code_explain(req: RunCodeRequest):
    out = []
    for i, line in enumerate(req.code.strip().split("\n"), 1):
        s = line.strip()
        if not s:
            continue
        d = f"Line {i}: "
        if s.startswith("#"):
            d += f"Comment - {s[1:].strip()}"
        elif s.startswith(("import ", "from ")):
            d += "Importing module"
        elif s.startswith("def "):
            d += f"Defines function '{s.split('(')[0].replace('def ', '')}'"
        elif s.startswith("class "):
            d += f"Defines class '{s.split('(')[0].replace('class ', '').split(':')[0]}'"
        elif s.startswith("return "):
            d += "Returns value"
        elif s.startswith("if "):
            d += "Conditional check"
        elif s.startswith("for "):
            d += "For loop"
        elif s.startswith("while "):
            d += "While loop"
        elif s.startswith("try:"):
            d += "Try block (error handling)"
        elif "=" in s and not s.startswith(("==", "!=", "<=", ">=")):
            d += "Variable assignment"
        else:
            d += "Statement"
        out.append(d)
    return {"total_lines": len(req.code.split("\n")), "explanation": out}


@app.post("/json_format")
async def json_format(req: JsonFormatRequest):
    try:
        p = json.loads(req.data)
        return {"valid": True, "formatted": json.dumps(p, indent=2, ensure_ascii=False),
                "type": type(p).__name__}
    except json.JSONDecodeError as ex:
        return {"valid": False, "error": str(ex), "formatted": None}


@app.post("/regex_test")
async def regex_test(req: RegexRequest):
    try:
        ms = list(re.finditer(req.pattern, req.text))
        return {"pattern": req.pattern, "match_count": len(ms),
                "matches": [{"match": m.group(), "start": m.start(),
                             "end": m.end(), "groups": m.groups()} for m in ms[:50]]}
    except re.error as ex:
        return {"error": str(ex), "matches": []}


@app.post("/pdf_extract")
async def pdf_extract(file: UploadFile = File(...)):
    try:
        import pdfplumber
        data = await file.read()
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, p in enumerate(pdf.pages):
                pages.append({"page": i + 1, "text": (p.extract_text() or "")[:2000]})
        return {"filename": file.filename, "pages": len(pages), "content": pages}
    except Exception as ex:
        return {"error": str(ex)}


@app.post("/ocr_image")
async def ocr_image(file: UploadFile = File(...)):
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(await file.read()))
        return {"filename": file.filename,
                "text": pytesseract.image_to_string(img).strip(),
                "size": f"{img.width}x{img.height}"}
    except Exception as ex:
        return {"error": str(ex)}


@app.post("/markdown_to_html")
async def markdown_to_html(req: MarkdownRequest):
    try:
        import markdown
        return {"html": markdown.markdown(req.text, extensions=["tables", "fenced_code"])}
    except Exception as ex:
        return {"error": str(ex)}


@app.post("/csv_to_json")
async def csv_to_json(req: CsvToJsonRequest):
    try:
        rd = csv.DictReader(io.StringIO(req.csv_data))
        rows = list(rd)
        return {"rows": len(rows), "columns": rd.fieldnames, "data": rows}
    except Exception as ex:
        return {"error": str(ex)}


@app.post("/url_shorten")
async def url_shorten(req: UrlShortenRequest):
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get("https://is.gd/create.php",
                            params={"format": "simple", "url": req.url})
            if r.status_code == 200:
                return {"original": req.url, "short": r.text.strip()}
            return {"error": "Shortening failed", "original": req.url}
        except Exception as ex:
            return {"error": str(ex)}


@app.post("/sentiment")
async def sentiment(req: SentimentRequest):
    try:
        from textblob import TextBlob
        bl = TextBlob(req.text)
        p = bl.sentiment.polarity
        label = "positive" if p > 0.1 else ("negative" if p < -0.1 else "neutral")
        return {"text_preview": req.text[:100], "label": label,
                "polarity": round(p, 3),
                "subjectivity": round(bl.sentiment.subjectivity, 3)}
    except Exception as ex:
        return {"error": str(ex)}


@app.post("/translate")
async def translate(req: TranslateRequest):
    try:
        from deep_translator import GoogleTranslator
        return {"original": req.text[:200],
                "translated": GoogleTranslator(source=req.source, target=req.target).translate(req.text),
                "source_lang": req.source, "target_lang": req.target}
    except Exception as ex:
        return {"error": str(ex)}


# ---------- browser-friendly GET wrappers ----------

@app.get("/test/search")
async def t_search(q: str = "artificial intelligence", max_results: int = 3):
    return await web_search(SearchRequest(query=q, max_results=max_results))

@app.get("/test/run")
async def t_run(code: str = "result = 2 + 2"):
    return await run_python(RunCodeRequest(code=code))

@app.get("/test/weather")
async def t_weather(city: str = "Delhi"):
    return await weather(WeatherRequest(city=city))

@app.get("/test/wiki")
async def t_wiki(topic: str = "Python programming language", sentences: int = 2):
    return await wikipedia_search(WikipediaRequest(topic=topic, sentences=sentences))

@app.get("/test/news")
async def t_news(topic: str = "technology", max_results: int = 3):
    return await news_headlines(NewsRequest(topic=topic, max_results=max_results))

@app.get("/test/sentiment")
async def t_sent(text: str = "I love this amazing tool"):
    return await sentiment(SentimentRequest(text=text))

@app.get("/test/translate")
async def t_trans(text: str = "Hello world", source: str = "en", target: str = "hi"):
    return await translate(TranslateRequest(text=text, source=source, target=target))

@app.get("/test/json")
async def t_json(data: str = '{"name":"test","value":42}'):
    return await json_format(JsonFormatRequest(data=data))

@app.get("/test/github")
async def t_gh(owner: str = "torvalds", repo: str = "linux"):
    return await github_repo(GithubRepoRequest(owner=owner, repo=repo))

@app.get("/test/arxiv")
async def t_arxiv(query: str = "large language models", max_results: int = 3):
    return await arxiv_search(ArxivRequest(query=query, max_results=max_results))


@app.get("/test")
async def test_page():
    return HTMLResponse("""<!DOCTYPE html><html><head><title>DSH Cloud Tools v2.2</title>
<style>body{font-family:sans-serif;max-width:820px;margin:40px auto;padding:20px;background:#1a1a2e;color:#eee}
.card{background:#16213e;padding:14px;margin:10px 0;border-radius:8px;border-left:4px solid #0f3460}
a{color:#4fc3f7;text-decoration:none}a:hover{text-decoration:underline}
h1{color:#4fc3f7}h3{margin:0 0 8px;color:#e94560}</style></head><body>
<h1>DSH Cloud Tools v2.2</h1><p>Pehli click ~30-60s (cold start), uske baad instant.</p>
<div class="card"><h3>Web Search</h3><a href="/test/search?q=artificial+intelligence&max_results=3" target="_blank">Search: artificial intelligence</a></div>
<div class="card"><h3>Python Run</h3><a href="/test/run?code=result%20%3D%202%20%2B%202" target="_blank">Run: result = 2 + 2</a></div>
<div class="card"><h3>Weather</h3><a href="/test/weather?city=Delhi" target="_blank">Weather in Delhi</a></div>
<div class="card"><h3>Wikipedia</h3><a href="/test/wiki?topic=Python+programming+language&sentences=2" target="_blank">Wikipedia: Python</a></div>
<div class="card"><h3>News</h3><a href="/test/news?topic=technology&max_results=3" target="_blank">Tech News</a></div>
<div class="card"><h3>arXiv</h3><a href="/test/arxiv?query=large+language+models&max_results=3" target="_blank">arXiv papers</a></div>
<div class="card"><h3>Sentiment</h3><a href="/test/sentiment?text=I+love+this+tool" target="_blank">Sentiment</a></div>
<div class="card"><h3>Translate</h3><a href="/test/translate?text=Hello+world&source=en&target=hi" target="_blank">EN to HI</a></div>
<div class="card"><h3>GitHub</h3><a href="/test/github?owner=torvalds&repo=linux" target="_blank">torvalds/linux</a></div>
<br><p style="color:#888"><a href="/health">/health</a> | <a href="/">/</a></p></body></html>""")

import os
import io
import json
import re
import csv
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from bs4 import BeautifulSoup
from contextlib import redirect_stdout, redirect_stderr

app = FastAPI(title="DSH Cloud Tools", version="2.1.0")

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_db():
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None


@app.on_event("startup")
def startup():
    conn = get_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS data_lake (
                    id SERIAL PRIMARY KEY,
                    source TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
            print("Database initialized")
        except Exception as e:
            print(f"DB init failed: {e}")
            conn.close()


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
    return {"status": "online", "version": "2.1.0", "tools": [
        "web_search", "fetch_url", "run_python", "text_process",
        "store_data", "query_data", "health",
        "wikipedia", "news", "weather", "arxiv",
        "github_repo", "code_explain", "json_format", "regex_test",
        "pdf_extract", "ocr_image", "markdown_to_html", "csv_to_json",
        "url_shorten", "sentiment", "translate"
    ]}

@app.get("/health")
async def health():
    db_ok = False
    conn = get_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            db_ok = True
        except Exception:
            pass
    return {"api": True, "database": db_ok}


@app.post("/search")
async def web_search(req: SearchRequest):
    results = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        try:
            resp = await client.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search",
                "srsearch": req.query, "srlimit": min(req.max_results, 5),
                "format": "json", "utf8": 1
            })
            if resp.status_code == 200:
                for item in resp.json().get("query", {}).get("search", []):
                    snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                    title = item.get("title", "")
                    results.append({
                        "title": title,
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "snippet": snippet[:300],
                        "source": "wikipedia"
                    })
        except Exception:
            pass
        try:
            resp = await client.get("https://api.duckduckgo.com/", params={
                "q": req.query, "format": "json", "no_redirect": 1
            })
            if resp.status_code == 200:
                ddg = resp.json()
                if ddg.get("AbstractText") and ddg.get("AbstractURL"):
                    results.append({
                        "title": ddg.get("Heading", req.query),
                        "url": ddg["AbstractURL"],
                        "snippet": ddg["AbstractText"][:300],
                        "source": "duckduckgo"
                    })
                for topic in ddg.get("RelatedTopics", [])[:req.max_results]:
                    if topic.get("Text") and topic.get("FirstURL"):
                        results.append({
                            "title": topic["Text"][:80],
                            "url": topic["FirstURL"],
                            "snippet": topic["Text"][:300],
                            "source": "duckduckgo"
                        })
                for r in ddg.get("Results", [])[:3]:
                    if r.get("Text") and r.get("FirstURL"):
                        results.append({
                            "title": r["Text"][:80],
                            "url": r["FirstURL"],
                            "snippet": r["Text"][:300],
                            "source": "duckduckgo"
                        })
        except Exception:
            pass
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    return {"query": req.query, "results": unique[:req.max_results], "total_found": len(unique)}


@app.post("/fetch")
async def fetch_url(url: str):
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return {"url": url, "status": resp.status_code, "content": text[:10000]}

@app.post("/run")
async def run_python(req: RunCodeRequest):
    output = io.StringIO()
    error = None
    result = None
    try:
        with redirect_stdout(output):
            with redirect_stderr(output):
                local_ns = {}
                exec(req.code, {"__builtins__": __builtins__}, local_ns)
                if "result" in local_ns:
                    result = str(local_ns["result"])
    except Exception as e:
        error = str(e)
    return {"output": output.getvalue(), "error": error, "result": result}

@app.post("/text")
async def text_process(req: TextProcessRequest):
    text = req.text
    op = req.operation
    if op == "summary":
        words = text.split()
        return {"word_count": len(words), "summary": " ".join(words[:100]) + ("..." if len(words) > 100 else "")}
    elif op == "words":
        return {"word_count": len(text.split()), "char_count": len(text)}
    elif op == "uppercase":
        return {"result": text.upper()}
    elif op == "lowercase":
        return {"result": text.lower()}
    elif op == "reverse":
        return {"result": text[::-1]}
    else:
        raise HTTPException(400, f"Unknown operation: {op}")

@app.post("/store")
async def store_data(req: StoreRequest):
    conn = get_db()
    if not conn:
        raise HTTPException(503, "Database not connected")
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO data_lake (source, content) VALUES (%s, %s) RETURNING id", (req.source, req.content))
        row_id = cur.fetchone()["id"]
        conn.commit()
        cur.close()
        conn.close()
        return {"stored": True, "id": row_id}
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))

@app.get("/query")
async def query_data(q: str, limit: int = 10):
    conn = get_db()
    if not conn:
        raise HTTPException(503, "Database not connected")
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, source, content, created_at FROM data_lake WHERE content ILIKE %s OR source ILIKE %s ORDER BY created_at DESC LIMIT %s", (f"%{q}%", f"%{q}%", limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"results": [dict(r) for r in rows], "count": len(rows)}
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))

@app.post("/wikipedia")
async def wikipedia_search(req: WikipediaRequest):
    try:
        import wikipedia
        wikipedia.set_lang("en")
        summary = wikipedia.summary(req.topic, sentences=req.sentences)
        page = wikipedia.page(req.topic)
        return {"topic": req.topic, "summary": summary, "url": page.url, "title": page.title}
    except Exception as e:
        return {"topic": req.topic, "error": str(e), "summary": ""}

@app.post("/news")
async def news_headlines(req: NewsRequest):
    try:
        import feedparser
        feeds = {
            "technology": "https://feeds.arstechnica.com/arstechnica/technology-lab",
            "science": "https://feeds.arstechnica.com/arstechnica/science",
            "world": "http://feeds.bbci.co.uk/news/world/rss.xml",
            "business": "http://feeds.bbci.co.uk/news/business/rss.xml",
            "general": "http://feeds.bbci.co.uk/news/rss.xml"
        }
        feed_url = feeds.get(req.topic.lower(), feeds["general"])
        parsed = feedparser.parse(feed_url)
        articles = []
        for entry in parsed.entries[:req.max_results]:
            articles.append({"title": entry.get("title", ""), "link": entry.get("link", ""), "published": entry.get("published", ""), "summary": entry.get("summary", "")[:300]})
        return {"topic": req.topic, "articles": articles}
    except Exception as e:
        return {"error": str(e), "articles": []}

@app.post("/weather")
async def weather(req: WeatherRequest):
    async with httpx.AsyncClient(timeout=15) as client:
        geo = await client.get(f"https://geocoding-api.open-meteo.com/v1/search?name={req.city}&count=1")
        geo_data = geo.json()
        if not geo_data.get("results"):
            return {"error": f"City '{req.city}' not found"}
        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        wx = await client.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto")
        wx_data = wx.json().get("current_weather", {})
        return {"city": loc["name"], "country": loc.get("country", ""), "temperature_c": wx_data.get("temperature"), "windspeed_kmh": wx_data.get("windspeed"), "weathercode": wx_data.get("weathercode"), "time": wx_data.get("time")}

@app.post("/arxiv")
async def arxiv_search(req: ArxivRequest):
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"http://export.arxiv.org/api/query?search_query=all:{req.query}&max_results={req.max_results}")
        soup = BeautifulSoup(resp.text, "xml")
        papers = []
        for e in soup.find_all("entry"):
            title = e.find("title").text.strip().replace("\n", " ") if e.find("title") else ""
            summary = e.find("summary").text.strip().replace("\n", " ")[:300] if e.find("summary") else ""
            link = e.find("id").text.strip() if e.find("id") else ""
            authors = [a.find("name").text for a in e.find_all("author") if a.find("name")]
            papers.append({"title": title, "authors": authors[:3], "summary": summary, "url": link})
        return {"query": req.query, "papers": papers}

@app.post("/github_repo")
async def github_repo(req: GithubRepoRequest):
    async with httpx.AsyncClient(timeout=15, headers={"Accept": "application/vnd.github.v3+json"}) as client:
        resp = await client.get(f"https://api.github.com/repos/{req.owner}/{req.repo}")
        if resp.status_code != 200:
            return {"error": f"Repo not found or rate limited ({resp.status_code})"}
        d = resp.json()
        return {"name": d.get("full_name"), "description": d.get("description"), "stars": d.get("stargazers_count"), "forks": d.get("forks_count"), "language": d.get("language"), "license": d.get("license", {}).get("spdx_id") if d.get("license") else None, "url": d.get("html_url"), "topics": d.get("topics", [])[:10]}

@app.post("/code_explain")
async def code_explain(req: RunCodeRequest):
    lines = req.code.strip().split("\n")
    explanation = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        desc = f"Line {i}: "
        if stripped.startswith("#"):
            desc += f"Comment - {stripped[1:].strip()}"
        elif stripped.startswith("import ") or stripped.startswith("from "):
            desc += "Importing module"
        elif stripped.startswith("def "):
            desc += f"Defines function '{stripped.split('(')[0].replace('def ', '')}'"
        elif stripped.startswith("class "):
            desc += f"Defines class '{stripped.split('(')[0].replace('class ', '').split(':')[0]}'"
        elif stripped.startswith("return "):
            desc += "Returns value"
        elif stripped.startswith("if "):
            desc += "Conditional check"
        elif stripped.startswith("for "):
            desc += "For loop"
        elif stripped.startswith("while "):
            desc += "While loop"
        elif stripped.startswith("try:"):
            desc += "Try block (error handling)"
        elif "=" in stripped and not any(stripped.startswith(kw) for kw in ["==", "!=", "<=", ">="]):
            desc += "Variable assignment"
        else:
            desc += "Statement"
        explanation.append(desc)
    return {"total_lines": len(lines), "explanation": explanation}

@app.post("/json_format")
async def json_format(req: JsonFormatRequest):
    try:
        parsed = json.loads(req.data)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        return {"valid": True, "formatted": formatted, "type": type(parsed).__name__}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e), "formatted": None}

@app.post("/regex_test")
async def regex_test(req: RegexRequest):
    try:
        matches = list(re.finditer(req.pattern, req.text))
        results = [{"match": m.group(), "start": m.start(), "end": m.end(), "groups": m.groups()} for m in matches]
        return {"pattern": req.pattern, "match_count": len(results), "matches": results[:50]}
    except re.error as e:
        return {"error": str(e), "matches": []}

@app.post("/pdf_extract")
async def pdf_extract(file: UploadFile = File(...)):
    try:
        import pdfplumber
        content = await file.read()
        pages_text = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_text.append({"page": i + 1, "text": text[:2000]})
        return {"filename": file.filename, "pages": len(pages_text), "content": pages_text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ocr_image")
async def ocr_image(file: UploadFile = File(...)):
    try:
        from PIL import Image
        import pytesseract
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img)
        return {"filename": file.filename, "text": text.strip(), "size": f"{img.width}x{img.height}"}
    except Exception as e:
        return {"error": str(e), "hint": "tesseract-ocr must be installed on server"}

@app.post("/markdown_to_html")
async def markdown_to_html(req: MarkdownRequest):
    try:
        import markdown
        html = markdown.markdown(req.text, extensions=["tables", "fenced_code", "codehilite"])
        return {"html": html}
    except Exception as e:
        return {"error": str(e)}

@app.post("/csv_to_json")
async def csv_to_json(req: CsvToJsonRequest):
    try:
        reader = csv.DictReader(io.StringIO(req.csv_data))
        rows = list(reader)
        return {"rows": len(rows), "columns": reader.fieldnames, "data": rows}
    except Exception as e:
        return {"error": str(e)}

@app.post("/url_shorten")
async def url_shorten(req: UrlShortenRequest):
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"https://is.gd/create.php?format=simple&url={req.url}")
            if resp.status_code == 200:
                return {"original": req.url, "short": resp.text.strip()}
            else:
                return {"error": "Shortening failed", "original": req.url}
        except Exception as e:
            return {"error": str(e)}

@app.post("/sentiment")
async def sentiment(req: SentimentRequest):
    try:
        from textblob import TextBlob
        blob = TextBlob(req.text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        label = "positive" if polarity > 0.1 else ("negative" if polarity < -0.1 else "neutral")
        return {"text_preview": req.text[:100], "label": label, "polarity": round(polarity, 3), "subjectivity": round(subjectivity, 3)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/translate")
async def translate(req: TranslateRequest):
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=req.source, target=req.target).translate(req.text)
        return {"original": req.text[:200], "translated": translated, "source_lang": req.source, "target_lang": req.target}
    except Exception as e:
        return {"error": str(e)}


@app.get("/test/search")
async def test_search(q: str = "AI news", max_results: int = 3):
    return await web_search(SearchRequest(query=q, max_results=max_results))

@app.get("/test/run")
async def test_run(code: str = "result = 2 + 2"):
    return await run_python(RunCodeRequest(code=code))

@app.get("/test/weather")
async def test_weather(city: str = "Delhi"):
    return await weather(WeatherRequest(city=city))

@app.get("/test/wiki")
async def test_wiki(topic: str = "Python programming language", sentences: int = 2):
    return await wikipedia_search(WikipediaRequest(topic=topic, sentences=sentences))

@app.get("/test/news")
async def test_news(topic: str = "technology", max_results: int = 3):
    return await news_headlines(NewsRequest(topic=topic, max_results=max_results))

@app.get("/test/sentiment")
async def test_sentiment(text: str = "I love this amazing tool"):
    return await sentiment(SentimentRequest(text=text))

@app.get("/test/translate")
async def test_translate(text: str = "Hello world", source: str = "en", target: str = "hi"):
    return await translate(TranslateRequest(text=text, source=source, target=target))

@app.get("/test/json")
async def test_json(data: str = '{"name":"test","value":42}'):
    return await json_format(JsonFormatRequest(data=data))

@app.get("/test/github")
async def test_github(owner: str = "torvalds", repo: str = "linux"):
    return await github_repo(GithubRepoRequest(owner=owner, repo=repo))

@app.get("/test")
async def test_page():
    html = """<!DOCTYPE html>
<html><head><title>DSH Cloud Tools v2.1</title>
<style>body{font-family:sans-serif;max-width:800px;margin:40px auto;padding:20px;background:#1a1a2e;color:#eee}
.card{background:#16213e;padding:15px;margin:10px 0;border-radius:8px;border-left:4px solid #0f3460}
a{color:#4fc3f7;text-decoration:none}a:hover{text-decoration:underline}
h1{color:#4fc3f7}h3{margin:0 0 8px 0;color:#e94560}</style></head>
<body>
<h1>DSH Cloud Tools v2.1 - Test Page</h1>
<p>Click any link. First click ~30-60s (cold start), then instant.</p>
<div class="card"><h3>Web Search (Wiki + DDG)</h3><a href="/test/search?q=artificial+intelligence&max_results=3" target="_blank">Search: artificial intelligence</a></div>
<div class="card"><h3>Python Run</h3><a href="/test/run?code=result%20%3D%202%20%2B%202" target="_blank">Run: result = 2 + 2</a></div>
<div class="card"><h3>Weather</h3><a href="/test/weather?city=Delhi" target="_blank">Weather in Delhi</a></div>
<div class="card"><h3>Wikipedia</h3><a href="/test/wiki?topic=Python+programming+language&sentences=2" target="_blank">Wikipedia: Python</a></div>
<div class="card"><h3>News</h3><a href="/test/news?topic=technology&max_results=3" target="_blank">Tech News</a></div>
<div class="card"><h3>Sentiment</h3><a href="/test/sentiment?text=I+love+this+tool" target="_blank">Sentiment analysis</a></div>
<div class="card"><h3>Translate</h3><a href="/test/translate?text=Hello+world&source=en&target=hi" target="_blank">EN to HI</a></div>
<div class="card"><h3>GitHub Repo</h3><a href="/test/github?owner=torvalds&repo=linux" target="_blank">torvalds/linux</a></div>
<br><p style="color:#888"><a href="/health">/health</a> | <a href="/">/</a></p>
</body></html>"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)

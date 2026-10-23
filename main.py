import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from bs4 import BeautifulSoup

app = FastAPI(title="DSH Cloud Tools", version="1.0.0")

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
            print("Database initialized successfully")
        except Exception as e:
            print(f"DB init failed (non-fatal): {e}")
            conn.close()
    else:
        print("No database configured - running without data lake")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class StoreRequest(BaseModel):
    source: str
    content: str


class RunCodeRequest(BaseModel):
    code: str


@app.get("/")
async def root():
    return {"status": "online", "tools": [
        "web_search", "fetch_url", "run_python",
        "store_data", "query_data", "text_process",
        "health"
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
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get("https://html.duckduckgo.com/html/", params={"q": req.query})
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for r in soup.select(".result"):
            title_el = r.select_one(".result__title a")
            snippet_el = r.select_one(".result__snippet")
            if title_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": title_el.get("href", ""),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else ""
                })
            if len(results) >= req.max_results:
                break
    return {"query": req.query, "results": results}


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
    import io
    import contextlib
    output = io.StringIO()
    error = None
    result = None
    try:
        with contextlib.redirect_stdout(output):
            with contextlib.redirect_stderr(output):
                local_ns = {}
                exec(req.code, {"__builtins__": __builtins__}, local_ns)
                if "result" in local_ns:
                    result = str(local_ns["result"])
    except Exception as e:
        error = str(e)
    return {"output": output.getvalue(), "error": error, "result": result}


@app.post("/store")
async def store_data(req: StoreRequest):
    conn = get_db()
    if not conn:
        raise HTTPException(503, "Database not connected")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO data_lake (source, content) VALUES (%s, %s) RETURNING id",
            (req.source, req.content)
        )
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
        cur.execute(
            "SELECT id, source, content, created_at FROM data_lake WHERE content ILIKE %s OR source ILIKE %s ORDER BY created_at DESC LIMIT %s",
            (f"%{q}%", f"%{q}%", limit)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"results": [dict(r) for r in rows], "count": len(rows)}
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.post("/text")
async def text_process(text: str, operation: str = "summary"):
    if operation == "summary":
        words = text.split()
        return {"original_length": len(text), "word_count": len(words),
                "summary": " ".join(words[:100]) + ("..." if len(words) > 100 else "")}
    elif operation == "words":
        return {"word_count": len(text.split()), "char_count": len(text)}
    elif operation == "uppercase":
        return {"result": text.upper()}
    elif operation == "lowercase":
        return {"result": text.lower()}
    elif operation == "reverse":
        return {"result": text[::-1]}
    else:
        raise HTTPException(400, f"Unknown operation: {operation}")

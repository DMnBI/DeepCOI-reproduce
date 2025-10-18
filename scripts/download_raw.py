import sys
import csv
import io
import requests

PREPROC = "https://portal.boldsystems.org/api/query/preprocessor"
QUERY   = "https://portal.boldsystems.org/api/query"
DOCS    = "https://portal.boldsystems.org/api/documents/{qid}/download"

TAXON  = sys.argv[1]
OUT    = f"{TAXON}.raw.tsv"

def get_query_id(norm_query: str, extent="full") -> str:
    r = requests.get(QUERY, params={"query": norm_query, "extent": extent}, timeout=60)
    r.raise_for_status()
    qid = r.json().get("query_id")
    if not qid:
        raise RuntimeError("No query_id from /api/query")
    return qid

def download_tsv(qid: str, out_path: str):
    url = DOCS.format(qid=qid)
    with requests.get(url, params={"format": "tsv"}, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in resp.iter_content(chunk_size=1<<20, decode_unicode=True):
                if chunk:
                    f.write(chunk)

if __name__ == "__main__":
    qid  = get_query_id(f"tax:phylum:{TAXON}", extent="full")
    print("query_id:", qid)
    download_tsv(qid, OUT)
    print("Saved:", OUT)
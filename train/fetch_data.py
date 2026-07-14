"""Download all public Poker44 benchmark releases into train/data/."""
import hashlib, json, pathlib, sys, time

import requests

BASE = "https://api.poker44.net/api/v1/benchmark"
OUT = pathlib.Path(__file__).parent / "data"


def main():
    OUT.mkdir(exist_ok=True)
    rel = requests.get(f"{BASE}/releases?limit=100", timeout=60).json()["data"]
    releases = rel.get("releases", rel if isinstance(rel, list) else [])
    attestation = []
    for r in releases:
        d = r["sourceDate"]
        f = OUT / f"{d}.json"
        if not f.exists():
            chunks, cursor = [], None
            while True:
                params = {"sourceDate": d, "limit": 24}
                if cursor:
                    params["cursor"] = cursor
                for attempt in range(3):
                    try:
                        data = requests.get(f"{BASE}/chunks", params=params, timeout=120).json()["data"]
                        break
                    except Exception as e:
                        print("retry", d, e); time.sleep(2)
                chunks.extend(data["chunks"])
                cursor = data.get("nextCursor")
                if not cursor:
                    break
            f.write_text(json.dumps(chunks))
            print(d, "->", len(chunks), "records")
        attestation.append({
            "sourceDate": d,
            "releaseVersion": r.get("releaseVersion"),
            "chunkCount": r.get("chunkCount"),
            "handCount": r.get("handCount"),
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        })
    att_path = pathlib.Path(__file__).parent.parent / "data_attestation.json"
    att_path.write_text(json.dumps({
        "statement": "Trained exclusively on the public Poker44 benchmark API "
                     "(https://api.poker44.net/api/v1/benchmark). No validator-private "
                     "data, no live labels, no other miners' artifacts.",
        "releases": attestation,
    }, indent=1))
    print("attestation ->", att_path)


if __name__ == "__main__":
    main()

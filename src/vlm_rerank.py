"""Gemini 3.5 Flash gated re-ranker: for crops the CLIP probe marks unknown, let the VLM
choose among the probe's OWN top-k candidates (or 'none'). It can never invent a brand —
the answer is constrained to candidates — so it recovers correct-but-shy brands without the
hallucination of open per-crop VLM. Parallel, capped, no-op when GEMINI_KEY is unset."""
import os, io, json, base64, urllib.request
from concurrent.futures import ThreadPoolExecutor

GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
MODEL = "gemini-2.5-flash-lite"   # lite is plenty for "pick 1 of 5 candidates": cheaper + higher free-tier quota + less contended than 3.5-flash (which 503s)
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"
MAX_CALLS = 12          # cap VLM calls per image (largest unknown crops first): stay under the
                        # free-tier ~15 req/min so one upload can't self-rate-limit; rest stay grey
WORKERS = 12
TIMEOUT = 30


def _b64(crop):
    buf = io.BytesIO(); crop.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


L1_CATS = ["飲料", "泡麵", "零食", "鮮食"]   # also ask category: VLM L1 beats the probe on "?" crops


def _ask(crop, candidates):
    prompt = ("This is ONE product cropped from a Taiwan convenience-store shelf. "
              "(1) Which of these candidate brands is it? Pick EXACTLY one, or \"none\" if none match "
              "or it is unreadable. Candidates: " + " / ".join(candidates) + ". "
              "(2) Which category: 飲料 (drink), 泡麵 (instant noodle), 零食 (snack), 鮮食 (fresh food)? "
              'Reply ONLY JSON: {"brand":"<one candidate or none>","cat":"飲料|泡麵|零食|鮮食"}')
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": _b64(crop)}},
                                     {"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    try:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
        d = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
        b = d.get("brand", "none"); c = d.get("cat", "")
        return (b if b in candidates else None,       # brand: hard-constrained to candidates
                c if c in L1_CATS else None, True)    # ok=True: Gemini answered (even if "none")
    except Exception:
        return (None, None, False)                    # transport/quota/parse failure -> Gemini unreachable


def rerank(crops, jobs):
    """jobs: list of (crop_index, [candidate brands]). Returns ({crop_index: {"brand","l1"}}, reachable):
    reachable=False when Gemini never answered (no key / quota / network) so callers can fall back."""
    if not GEMINI_KEY or not jobs:
        return {}, False
    jobs = sorted(jobs, key=lambda j: -(crops[j[0]].size[0]*crops[j[0]].size[1]))[:MAX_CALLS]
    out = {}
    reachable = False
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_ask, crops[i], cands): i for i, cands in jobs}
        for f in futs:
            brand, l1, ok = f.result()
            reachable = reachable or ok
            if brand or l1:
                out[futs[f]] = {"brand": brand, "l1": l1}
    return out, reachable

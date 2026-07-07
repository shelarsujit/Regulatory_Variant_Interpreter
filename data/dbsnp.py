"""Resolve dbSNP rsIDs to (ref, alt) alleles via the myvariant.info batch API.

THEORY (plain English):
The Deng variant table (Data S2) identifies each tested SNV by rsID + hg38 position but does
NOT carry the ref/alt bases (docs/01_data_provenance §4). We need those bases to build the ALT
sequence the model scores. myvariant.info is a high-throughput variant-annotation service: one
POST resolves up to 1000 rsIDs, returning each SNP's dbSNP `ref`/`alt`.

WHY myvariant, not Ensembl REST: Ensembl's variation endpoint is unreliable at this scale
(200-id batches time out, intermittent HTTP 500), turning a 15k-variant run into an hour of
backoff. myvariant answers ~1000 IDs in ~1 s. It returns one row PER allele, so multi-allelic
sites yield several rows we group back by rsID (ref is shared; we collect all alts). Its `_id`
coordinates are hg19-anchored, but we only take the allele LETTERS (assembly-invariant for the
overwhelming majority of SNVs); positioning uses Deng's own hg38 `variant_pos`, and load_deng's
ref-vs-hg38 concordance check (+ reverse-complement rescue) catches the rare disagreement.

Batched and cached to disk, so a full ~15k run makes ~16 requests once, then reads the cache on
every rerun. Cache format: {rsid: {"ref": str, "alts": [str, ...]}} or {rsid: None}.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

_MYVARIANT = "https://myvariant.info/v1/query"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CACHE = os.path.join(_HERE, "raw", "dbsnp_cache.json")


def _load_cache(path):
    if path and os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def _post_batch(ids, timeout, retries=4):
    """POST one batch of rsIDs to myvariant; return the raw list of hit dicts."""
    data = urllib.parse.urlencode({
        "q": ",".join(ids), "scopes": "dbsnp.rsid", "fields": "dbsnp.ref,dbsnp.alt",
    }).encode()
    req = urllib.request.Request(_MYVARIANT, data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded",
                                          "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            transient = e.code == 429 or 500 <= e.code < 600
            if transient and attempt < retries - 1:
                time.sleep((float(e.headers.get("Retry-After", 0)) or 2 ** attempt) + 1)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return []


def _group_hits(hits):
    """Fold myvariant's per-allele rows into {rsid: {"ref", "alts"}} (SNVs only)."""
    grouped: dict[str, dict] = {}
    for h in hits:
        q = h.get("query")
        if not q or h.get("notfound"):
            continue
        db = h.get("dbsnp")
        if not isinstance(db, dict):
            continue
        ref, alt = db.get("ref"), db.get("alt")
        if not ref or not alt or set(str(ref)) - set("ACGTN") or set(str(alt)) - set("ACGTN"):
            continue                                 # skip indels / non-ACGT
        g = grouped.setdefault(q, {"ref": ref, "alts": []})
        if alt not in g["alts"]:
            g["alts"].append(alt)
    return grouped


def resolve_rsids(rsids, *, cache_path: str = _DEFAULT_CACHE, batch_size: int = 1000,
                  timeout: int = 60, sleep_between: float = 0.2, verbose: bool = True) -> dict:
    """Map each rsID -> {"ref": str, "alts": [str, ...]} (SNV alleles only), or None.

    Uses an on-disk cache; only uncached IDs hit the network. Failed batches are requeued on
    later sweeps; the cache is checkpointed after each batch so a run resumes where it stopped.
    """
    unique = list(dict.fromkeys(str(r) for r in rsids if r and str(r) != "nan"))
    cache = _load_cache(cache_path)

    max_sweeps = 6
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        todo = [r for r in unique if r not in cache]
        if not todo:
            break
        n_batches = -(-len(todo) // batch_size)
        if verbose:
            print(f"[dbsnp] sweep {sweep}: {len(todo)} uncached rsIDs "
                  f"({len(unique) - len(todo)} done) in {n_batches} batches of {batch_size}...")
        progressed = False
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            try:
                grouped = _group_hits(_post_batch(batch, timeout))
            except Exception as e:
                if verbose:
                    print(f"[dbsnp]   batch {i//batch_size} failed ({type(e).__name__}: "
                          f"{str(e)[:100]}); will retry next sweep")
                continue
            for rsid in batch:
                cache[rsid] = grouped.get(rsid)      # None when unresolved (won't re-query)
            _save_cache(cache, cache_path)
            progressed = True
            if verbose:
                done = sum(1 for r in unique if r in cache)
                print(f"[dbsnp]   {done}/{len(unique)} resolved")
            time.sleep(sleep_between)
        if not progressed:
            if verbose:
                print(f"[dbsnp] sweep {sweep} made no progress — service unavailable; "
                      "stopping (rerun later to resume from cache)")
            break

    still_missing = sum(1 for r in unique if r not in cache)
    if verbose and still_missing:
        print(f"[dbsnp] {still_missing}/{len(unique)} rsIDs still unresolved after {sweep} sweeps")
    return {r: cache.get(r) for r in unique}

"""Build the context module's data, offline, once: the corpus, the int8 embeddings, the reranker
scores, the PCA projection and the manifest (src/agent_loop_sim/data/context/).

    pip install -e ".[offline]"
    python scripts/build_context_data.py --cache ~/.cache/agent-context-offline

Needs the network the first time (the SQuAD file and two ONNX models from the Hugging Face Hub,
pinned to the revisions below) and runs on CPU. Nothing in CI or on the sites runs this: they
read the files it writes, and the engine recomputes every metric from the int8 vectors. The
float32 numbers in manifest.json["offline_float32"] are the only ones that come from this run
alone (they need the unquantised vectors, which are not shipped).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import sys
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_sim import VERSION  # noqa: E402
from agent_loop_sim.context.corpus import CHUNKERS, DEFAULT, Corpus, spans_sha256  # noqa: E402
from agent_loop_sim.context.evaluate import KS, mean_metrics, metrics, relevant  # noqa: E402
from agent_loop_sim.rng import Rng  # noqa: E402
from agent_loop_sim.tokenizer import default_tokenizer  # noqa: E402

OUT = ROOT / "src/agent_loop_sim/data/context"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
ARTICLES = ["Computational_complexity_theory", "Packet_switching", "Steam_engine", "Oxygen", "Prime_number",
            "Apollo_program"]
N_QUESTIONS = 200
SAMPLE_SEED = 23
EMBEDDER = {"repo": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "file": "onnx/model.onnx", "licence": "Apache-2.0", "max_tokens": 256, "dim": 384,
            "pooling": "mean over the attention mask, then L2-normalised"}
RERANKER = {"repo": "cross-encoder/ms-marco-MiniLM-L6-v2", "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            "file": "onnx/model.onnx", "licence": "Apache-2.0", "max_tokens": 512,
            "score": "the single output logit, x 1000, rounded to an integer"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_corpus(squad: dict) -> dict:
    by_title = {a["title"]: a for a in squad["data"]}
    articles = []
    pool = []
    for ai, title in enumerate(ARTICLES):
        a = by_title[title]
        text = ""
        for pi, p in enumerate(a["paragraphs"]):
            ctx = p["context"].replace("\n", " ")  # same length: answer offsets still hold
            if unicodedata.normalize("NFC", ctx) != ctx or any(ord(ch) > 0xFFFF for ch in ctx):
                raise SystemExit("corpus text must be NFC without astral characters (UTF-16 offsets = code points)")
            if pi:
                text += "\n\n"
            off = len(text)
            text += ctx
            for q in p["qas"]:
                ans = q["answers"][0]
                st = off + ans["answer_start"]
                en = st + len(ans["text"])
                assert text[st:en] == ans["text"]
                pool.append({"id": q["id"], "article": ai, "question": q["question"].strip(), "answer": ans["text"],
                             "start": st, "end": en})
        articles.append({"title": title.replace("_", " "), "source_title": title, "text": text})
    # a seeded partial Fisher-Yates shuffle picks the questions; they are kept in corpus order
    idx = list(range(len(pool)))
    r = Rng(SAMPLE_SEED)
    for i in range(N_QUESTIONS):
        j = i + int(r.random() * (len(idx) - i))
        idx[i], idx[j] = idx[j], idx[i]
    chosen = sorted(idx[:N_QUESTIONS])
    return {"source": "SQuAD v1.1 development set", "url": SQUAD_URL, "sha256": SQUAD_SHA256,
            "licence": "CC BY-SA 4.0", "articles": articles, "questions": [pool[i] for i in chosen],
            "pool": len(pool)}


class Onnx:
    def __init__(self, cache: Path, spec: dict) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.model_path = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"], cache_dir=cache / "hf"))
        self.tok_path = Path(hf_hub_download(spec["repo"], "tokenizer.json", revision=spec["revision"], cache_dir=cache / "hf"))
        self.tok = Tokenizer.from_file(str(self.tok_path))
        self.tok.enable_truncation(spec["max_tokens"])
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(self.model_path), so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.max_tokens = spec["max_tokens"]

    def feed(self, encs):
        import numpy as np

        f = {"input_ids": np.array([e.ids for e in encs], dtype=np.int64),
             "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
             "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64)}
        return {k: v for k, v in f.items() if k in self.inputs}, f["attention_mask"]

    def embed(self, texts: list[str], batch: int = 16):
        import numpy as np

        out = []
        for i in range(0, len(texts), batch):
            encs = self.tok.encode_batch(texts[i:i + batch])
            feed, mask = self.feed(encs)
            h = self.sess.run(None, feed)[0]
            m = mask[:, :, None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            out.append(v.astype(np.float32))
        return np.concatenate(out)

    def n_truncated(self, texts: list[str]) -> int:
        self.tok.no_truncation()
        self.tok.no_padding()
        n = sum(1 for e in self.tok.encode_batch(texts) if len(e.ids) > self.max_tokens)
        self.tok.enable_truncation(self.max_tokens)
        self.tok.enable_padding()
        return n

    def score_pairs(self, pairs: list[tuple[str, str]], batch: int = 16):
        """Scores in input order; batches are formed by length so little is padded."""
        import numpy as np

        lens = [len(e.ids) for e in self._raw(pairs)]
        order = sorted(range(len(pairs)), key=lambda i: lens[i])
        out = np.zeros(len(pairs), dtype=np.float32)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            encs = self.tok.encode_batch([pairs[i] for i in idx])
            feed, _ = self.feed(encs)
            out[idx] = self.sess.run(None, feed)[0][:, 0]
            if (b // batch) % 50 == 0:
                print(f"  scored {b + len(idx)} of {len(pairs)}", flush=True)
        return out

    def _raw(self, items):
        self.tok.no_padding()
        encs = self.tok.encode_batch(items)
        self.tok.enable_padding()
        return encs


POOL = "per question: the top 30 of RRF (k 60, depth 50) with the BM25 top 10 and the dense (int8) top 10"


def rerank_pools(c: Corpus, files: dict) -> list[list[int]]:
    """The chunks the cross-encoder scores for each question, from the engine's own rankings of the
    shipped int8 vectors (so the engine can rerank any of these lists up to its depth)."""
    from agent_loop_sim.context.retrieval import Retriever, rank, rrf

    r = Retriever(c, DEFAULT)
    out = []
    for qi in range(len(c.questions)):
        rb = rank(r.bm25_scores(qi))
        rd = rank(r.dense(qi))
        fused = rank(rrf([rb, rd], len(r.chunks)))
        pool: list[int] = []
        for ch in fused[:30] + rb[:10] + rd[:10]:
            if ch not in pool:
                pool.append(ch)
        out.append(pool)
    return out


def quantise(v) -> list[list[int]]:
    import numpy as np

    s = 127.0 / np.abs(v).max(axis=1, keepdims=True)
    q = np.clip(np.rint(v * s), -127, 127).astype(np.int8)
    return q


def emb_file(name: str, q, extra: dict) -> dict:
    return dict({"set": name, "model": EMBEDDER["repo"], "revision": EMBEDDER["revision"], "dim": int(q.shape[1]),
                 "n": int(q.shape[0]), "scale": "per vector, largest component to +-127, rounded"}, **extra,
                b64=base64.b64encode(q.tobytes()).decode("ascii"))


def write(name: str, obj: dict) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {name} ({(OUT / name).stat().st_size} bytes)", flush=True)


def main() -> None:
    import numpy as np
    import onnxruntime

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--reuse", action="store_true", help="reuse float embeddings saved by an earlier run (same model and texts)")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sq = a.cache / "dev-v1.1.json"
    if not sq.exists():
        urllib.request.urlretrieve(SQUAD_URL, sq)
    if sha256_file(sq) != SQUAD_SHA256:
        raise SystemExit("SQuAD file checksum mismatch")
    corpus = build_corpus(json.loads(sq.read_text(encoding="utf-8")))
    write("corpus.json", corpus)
    files: dict = {"corpus.json": corpus}
    c = Corpus(corpus, files, default_tokenizer())
    emb = Onnx(a.cache, EMBEDDER)

    sents = c.sentences()
    s_texts = [corpus["articles"][ai]["text"][st:en].strip() for ai, st, en in sents]
    cache_npz = a.cache / "floats.npz"
    saved = dict(np.load(cache_npz)) if (a.reuse and cache_npz.exists()) else {}

    def embed(key: str, texts: list[str]):
        if key not in saved:
            saved[key] = emb.embed(texts)
            np.savez(cache_npz, **saved)
        return saved[key]

    sv = embed("sentences", s_texts)
    files["emb-sentences.json"] = emb_file("sentences", quantise(sv), {"truncated": emb.n_truncated(s_texts)})
    write("emb-sentences.json", files["emb-sentences.json"])

    q_texts = [q["question"] for q in corpus["questions"]]
    qv = embed("questions", q_texts)
    files["emb-questions.json"] = emb_file("questions", quantise(qv), {"truncated": emb.n_truncated(q_texts)})
    write("emb-questions.json", files["emb-questions.json"])

    float_metrics = {}
    chunk_float = {}
    for name in CHUNKERS:
        chunks = c.chunks(name)
        texts = [c.chunk_text(ch) for ch in chunks]
        v = embed(name, texts)
        chunk_float[name] = v
        sha = spans_sha256([[ch["article"], ch["start"], ch["end"]] for ch in chunks])
        files[f"emb-{name}.json"] = emb_file(name, quantise(v), {"spans_sha256": sha, "truncated": emb.n_truncated(texts)})
        write(f"emb-{name}.json", files[f"emb-{name}.json"])
        rows = []
        for qi, q in enumerate(corpus["questions"]):
            s = v @ qv[qi]
            order = sorted(range(len(chunks)), key=lambda i: (-float(s[i]), i))
            rows.append(metrics(order, relevant(chunks, q)))
        float_metrics[name] = mean_metrics(rows)

    emb_paths = (emb.model_path, emb.tok_path)
    del emb
    rr = Onnx(a.cache, RERANKER)
    pools = rerank_pools(c, files)
    pairs = [(q_texts[qi], c.chunk_text(c.chunks(DEFAULT)[ch])) for qi, pool in enumerate(pools) for ch in pool]
    print(f"reranking {len(pairs)} pairs (pool: {POOL})", flush=True)
    sc = rr.score_pairs(pairs)
    q16 = np.clip(np.rint(sc * 1000), -32768, 32767).astype(int)
    scores, k = [], 0
    for pool in pools:
        scores.append([[ch, int(q16[k + j])] for j, ch in enumerate(pool)])
        k += len(pool)
    files["rerank.json"] = {"model": RERANKER["repo"], "revision": RERANKER["revision"], "config": DEFAULT,
                            "pool": POOL, "scale": 1000, "pairs": len(pairs), "scores": scores}
    write("rerank.json", files["rerank.json"])

    # PCA of the default config's float chunk vectors; questions projected onto the same axes
    X = chunk_float[DEFAULT]
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    axes = vt[:2]
    for k in range(2):  # a fixed sign: the largest-magnitude loading positive
        if axes[k][np.argmax(np.abs(axes[k]))] < 0:
            axes[k] = -axes[k]
    pc = (X - mu) @ axes.T
    pq = (qv - mu) @ axes.T
    var = (np.var((X - mu) @ vt.T, axis=0))
    files["pca.json"] = {"config": DEFAULT, "explained": [round(float(x), 4) for x in (var[:2] / var.sum())],
                         "chunks": [[round(float(x), 4), round(float(y), 4)] for x, y in pc],
                         "questions": [[round(float(x), 4), round(float(y), 4)] for x, y in pq]}
    write("pca.json", files["pca.json"])

    manifest = {
        "engine": VERSION,
        "corpus": {"source": corpus["source"], "url": SQUAD_URL, "sha256": SQUAD_SHA256, "licence": corpus["licence"],
                   "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                   "citation": "Rajpurkar, Zhang, Lopyrev and Liang, SQuAD: 100,000+ Questions for Machine Comprehension of Text, EMNLP 2016 (arXiv:1606.05250)",
                   "articles": ARTICLES, "questions": N_QUESTIONS, "question_pool": corpus["pool"], "sample_seed": SAMPLE_SEED,
                   "changes": "paragraph newlines replaced by spaces; paragraphs joined by a blank line; first gold answer kept"},
        "embedder": dict(EMBEDDER, onnx_sha256=sha256_file(emb_paths[0]), tokenizer_sha256=sha256_file(emb_paths[1])),
        "reranker": dict(RERANKER),
        "runtime": {"onnxruntime": onnxruntime.__version__, "numpy": np.__version__, "python": platform.python_version(),
                    "machine": platform.machine(), "threads": 2},
        "offline_float32": float_metrics,
        "files": {},
    }
    manifest["reranker"]["onnx_sha256"] = sha256_file(rr.model_path)
    manifest["reranker"]["tokenizer_sha256"] = sha256_file(rr.tok_path)
    for n in sorted(files):
        manifest["files"][n] = sha256_file(OUT / n)
    write("manifest.json", manifest)
    print("float32 dense recall@5:", {k: round(v["recall@5"], 4) for k, v in float_metrics.items()}, KS)


if __name__ == "__main__":
    main()

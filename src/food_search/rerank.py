"""Reranking of the fused candidate pool.

Two interchangeable rerankers:
  * LLMReranker - listwise gpt-4.1-mini; one call per query sees all
    candidates jointly. Falls back to the fused order on any failure.
  * CrossEncoderReranker - local pointwise cross-encoder (mMARCO MiniLM or
    bge-reranker-v2-m3). Zero API cost and no network latency at query time;
    the distillation literature (Rank-DistiLLM, LiT5) shows small
    cross-encoders can match LLM rankers on short passages.
"""

from __future__ import annotations

import json

from .retrieval import SearchResult

# Short aliases -> Hugging Face model ids.
CROSS_ENCODERS = {
    "minilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", # 118M, trained on mMARCO (incl. PT)
    "bge": "BAAI/bge-reranker-v2-m3", # 568M multilingual, stronger but heavier
}

SYSTEM_PROMPT = (
    "You rank food-delivery catalog items for a Portuguese search query. "
    "Judge how well each item satisfies the query's intent: dish type, preparation "
    "style, cuisine and key attributes. Items that are not food at all are never relevant. "
    'Respond with JSON only: {"ranking": [<id>, ...]} listing ALL given ids from most '
    "to least relevant."
)


class LLMReranker:
    def __init__(self, model: str = "gpt-4.1-mini"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        # compact candidate cards; ids are list positions
        payload = [
            {
                "id": i,
                "name": r.item.name,
                "description": r.item.description[:200],
                "category": r.item.category,
                "type": r.item.taxonomy,
            }
            for i, r in enumerate(candidates)
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nItems:\n{json.dumps(payload, ensure_ascii=False)}",
                    },
                ],
            )
            order = json.loads(response.choices[0].message.content)["ranking"]
            # the model may repeat or invent ids - keep valid first-seen only
            seen: set[int] = set()
            ranked = [
                candidates[i]
                for i in order
                if isinstance(i, int) and 0 <= i < len(candidates) and not (i in seen or seen.add(i))
            ]
            # Append anything the model dropped, preserving fused order.
            ranked.extend(c for i, c in enumerate(candidates) if i not in seen)
            return ranked
        except Exception as exc: # noqa: BLE001 - degrade gracefully to fused order
            print(f"  [rerank] falling back to fused order for {query!r}: {exc}")
            return candidates


class CrossEncoderReranker:
    def __init__(
        self,
        alias: str = "minilm",
        batch_size: int = 32,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = CROSS_ENCODERS.get(alias, alias)
        self.model = CrossEncoder(self.model_name)
        self.batch_size = batch_size

    @staticmethod
    def _passage(result: SearchResult) -> str:
        # same fields the LLM reranker sees, to keep the comparison fair
        item = result.item
        parts = [item.name]
        if item.description:
            parts.append(item.description[:200])
        if item.category:
            parts.append(item.category)
        return ". ".join(parts)

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        pairs = [(query, self._passage(c)) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [candidates[i] for i in order]

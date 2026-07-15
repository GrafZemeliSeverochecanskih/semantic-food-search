"""Load the catalog and queries, and build one searchable document per item.

Each catalog row carries two JSON blobs:
  * itemMetadata - name, description, category, taxonomy, dietary flags, price
  * itemProfile  - behavioral data, including the real search terms users typed
                   before buying the item (a strong relevance signal we fold
                   into the document text).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Small Portuguese stopword list for BM25 tokenization. Deliberately short:
# over-aggressive stopword removal hurts short food names ("pão de queijo").
PT_STOPWORDS = frozenset(
    "a o e de da do das dos em na no nas nos um uma uns umas com para por que ao aos à às".split()
)


@dataclass
class Item:
    item_id: str
    name: str
    description: str
    category: str
    taxonomy: str
    price: float | None
    merchant_id: str
    dietary: list[str] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    image: str | None = None

    @property
    def image_url(self) -> str | None:
        if not self.image:
            return None
        return f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{self.image}"

    def document(self) -> str:
        """Text used for both BM25 and dense indexing.

        Field order puts the highest-signal text (name) first; user search
        terms act as query-language expansion of the catalog language.
        """
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.category:
            parts.append(f"Categoria: {self.category}")
        if self.taxonomy:
            parts.append(f"Tipo: {self.taxonomy}")
        if self.dietary:
            parts.append(" ".join(self.dietary))
        if self.search_terms:
            parts.append("Buscado como: " + ", ".join(self.search_terms))
        return ". ".join(parts)


def load_items(csv_path: Path) -> list[Item]:
    df = pd.read_csv(csv_path)
    # The raw file contains 3 duplicate itemIds; keep the first occurrence.
    df = df.drop_duplicates(subset="itemId", keep="first")
    items: list[Item] = []
    for row in df.itertuples(index=False):
        meta = json.loads(row.itemMetadata)
        try:
            profile = json.loads(row.itemProfile) if isinstance(row.itemProfile, str) else {}
        except json.JSONDecodeError:
            profile = {}

        # "OUTROS" levels carry no signal, skip them
        taxonomy = meta.get("taxonomy") or {}
        taxonomy_str = " > ".join(
            v.replace("_", " ").title()
            for v in (taxonomy.get("l0"), taxonomy.get("l1"), taxonomy.get("l2"))
            if v and v != "OUTROS"
        )
        # expose boolean flags as Portuguese words so both indexes can match them
        dietary = [
            label
            for flag, label in (("vegan", "vegano"), ("organic", "orgânico"), ("lacFree", "sem lactose"))
            if meta.get(flag)
        ]
        # Real user search terms, most frequent first, deduped, capped at 8.
        terms = sorted(profile.get("search") or [], key=lambda t: -t.get("count", 0))
        seen: set[str] = set()
        search_terms = []
        for t in terms:
            term = (t.get("term") or "").strip().lower()
            if term and term not in seen:
                seen.add(term)
                search_terms.append(term)
            if len(search_terms) == 8:
                break

        images = meta.get("images") or []
        items.append(
            Item(
                item_id=str(row.itemId),
                name=meta.get("name") or "",
                description=(meta.get("description") or "").strip(),
                category=meta.get("category_name") or "",
                taxonomy=taxonomy_str,
                price=meta.get("price"),
                merchant_id=str(row.merchantId),
                dietary=dietary,
                search_terms=search_terms,
                image=images[0] if images else None,
            )
        )
    return items


def load_queries(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path)["search_term_pt"].dropna().astype(str).tolist()


def normalize(text: str) -> str:
    """Lowercase and strip accents so 'Pão' matches 'pao'."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Word tokens for BM25: accent-free, lowercased, minus stopwords."""
    return [t for t in re.findall(r"\w+", normalize(text)) if t not in PT_STOPWORDS]

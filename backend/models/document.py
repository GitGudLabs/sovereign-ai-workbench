from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Page:
    page_number: int
    text: str

@dataclass
class Document:
    id: str
    filename: str
    pages: List[Page] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

"""RAG 파이프라인 모듈 묶음.

여기서 하위 모듈을 eager import 하지 않는다. `vector_store` 는 faiss,
`embeddings` 는 langchain_huggingface 를 필요로 하는데, 패키지를 import 하는
것만으로 그 무거운/미설치 의존성이 끌려오면 loader/normalizer 만 쓰려는
테스트까지 함께 죽는다. 필요한 모듈을 직접 import 해서 쓴다.

    from petcare_ai.rag.loader import load_documents
    from petcare_ai.rag.normalizer import normalize_documents
"""

from __future__ import annotations

__all__: list[str] = []

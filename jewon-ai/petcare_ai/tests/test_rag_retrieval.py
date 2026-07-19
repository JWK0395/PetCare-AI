"""RAG 검색 계층 테스트 (명세 11·12·13절 + 17절 검색 테스트).

검증 대상은 네 모듈이 함께 지키는 **하나의 검색 계약**이다.

- `rag/embeddings.py` — 오프라인 임베딩이 결정적이고 정규화돼 있으며 빈 입력에도 안전하다.
- `rag/vector_store.py` — build→save→load 왕복이 성립하고, 설정이 바뀌면 옛 index 를
  절대 재사용하지 않으며(`load()=False`), dog/cat index 가 구조적으로 분리돼 있다.
- `rag/query_builder.py` — LLM 없이도 한국어·영어 query 를 만들고, LLM 이 주어지면
  structured output 으로 개선하되 실패하면 규칙 기반으로 되돌아간다.
- `rag/retriever.py` — ko/en 결과를 합치고 chunk_id 중복을 제거하며(더 높은 score 유지),
  `query.species` 이외의 index 는 건드리지 않는다.

원칙:
  * 실제 임베딩 모델(bge-m3/e5) 로드 없음 — `DeterministicEmbeddings` 만 사용.
  * 네트워크 호출 없음 — LLM 은 전부 stub 주입.
  * `rag.chunker` 를 import 하지 않는다 — langchain_text_splitters → torch 가 끌려와
    테스트가 수십 초 느려진다. 대신 "문서 1건 = chunk 1건" 더미를 직접 만든다
    (vector store 는 chunk 를 속성/dict 어느 쪽으로도 읽으므로 계약이 동일하다).
  * 파일 산출물은 전부 `tmp_path` 안에만 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from petcare_ai.config import RagSettings, Settings, Species
from petcare_ai.rag.embeddings import (
    DETERMINISTIC_DIM,
    DeterministicEmbeddings,
    build_embeddings,
    embedding_dimension,
)
from petcare_ai.rag.query_builder import (
    SENIOR_AGE_YEARS,
    build_rag_query,
    build_rag_query_rule_based,
)
from petcare_ai.rag.retriever import (
    MAX_FINAL_EVIDENCE,
    MIN_FINAL_EVIDENCE,
    deduplicate_evidence,
    resolve_final_evidence_limit,
    retrieve,
)
from petcare_ai.rag.vector_store import VeterinaryVectorStore
from petcare_ai.schemas import RagQuery, RetrievedEvidence

#: 한글 음절 영역 — "영어 query 에 한국어가 섞이지 않았는지" 판정에 쓴다.
_HANGUL_RANGE = ("가", "힣")


def _has_hangul(text: str) -> bool:
    """문자열에 한글 음절이 하나라도 있는지 — 영어 query 검증용."""
    return any(_HANGUL_RANGE[0] <= ch <= _HANGUL_RANGE[1] for ch in text)


# ---------------------------------------------------------------------------
# 더미 chunk — chunker 를 import 하지 않기 위한 최소 구현
# ---------------------------------------------------------------------------
@dataclass
class FakeChunk:
    """`chunker.Chunk` 와 동일한 4개 필드만 가진 테스트용 chunk.

    vector store 는 `chunk_id / document_id / text / metadata` 만 읽으므로
    실제 `Chunk`(pydantic) 없이도 계약이 완전히 동일하다. chunker 를 import 하면
    langchain_text_splitters → torch 가 끌려와 테스트가 느려지므로 의도적으로 피한다.
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def make_chunk(
    document_id: str,
    species: Species,
    title: str,
    text: str,
    categories: list[str] | None = None,
    heading_path: list[str] | None = None,
) -> FakeChunk:
    """문서 1건 = chunk 1건으로 더미 chunk 를 만든다.

    metadata 키는 `chunker._build_metadata()` 가 채우는 명세 10절 키를 그대로 따른다.
    vector store 가 metadata 를 `RetrievedEvidence` 로 옮길 때 어떤 키를 보는지가
    이 테스트의 검증 대상이기 때문에, 키 이름을 임의로 줄이지 않는다.
    """
    chunk_id = f"{document_id}#0"
    return FakeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        metadata={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "species": species,
            "title": title,
            "source": (
                "Cornell Riney Canine Health Center"
                if species == "dog"
                else "Cornell Feline Health Center"
            ),
            "source_url": f"https://www.vet.cornell.edu/{species}/{document_id}",
            "categories": list(categories or []),
            "last_updated": "2024-01-01",
            "medical_domain": "canine_health" if species == "dog" else "feline_health",
            "language": "en",
            "content_hash": f"hash-{document_id}",
            "heading_path": list(heading_path or []),
        },
    )


#: 강아지 문서 10건. retriever 의 상한(final_evidence_max) 절단이 실제로 일어나도록
#: index 크기를 반환 상한보다 크게 잡는다.
DOG_CHUNKS: list[FakeChunk] = [
    make_chunk(
        "dog-vomit",
        "dog",
        "Vomiting in Dogs",
        "Vomiting in dogs can follow dietary indiscretion, infection, or intestinal "
        "obstruction. Red flags include repeated vomiting, blood in the vomit, and "
        "lethargy in a dog that will not eat.",
        categories=["Digestive system", "Emergency"],
        heading_path=["Digestive problems", "Vomiting"],
    ),
    make_chunk(
        "dog-cough",
        "dog",
        "Coughing in Dogs",
        "Cough in dogs may come from tracheal collapse, kennel cough, or heart disease. "
        "An older dog with a heart murmur that coughs at night needs a cardiac workup.",
        categories=["Respiratory system", "Heart"],
        heading_path=["Respiratory problems", "Cough"],
    ),
    make_chunk(
        "dog-heart",
        "dog",
        "Mitral Valve Disease in Dogs",
        "Mitral valve disease is the most common acquired heart disease of older dogs. "
        "Owner observations of resting respiratory rate help detect congestive heart failure.",
        categories=["Heart", "Senior care"],
        heading_path=["Cardiology", "Mitral valve disease"],
    ),
    make_chunk(
        "dog-diarrhea",
        "dog",
        "Diarrhea in Dogs",
        "Diarrhea in dogs is often self limiting, but bloody diarrhea with lethargy is an "
        "emergency. Owners should record stool frequency and appetite for the veterinarian.",
        categories=["Digestive system"],
        heading_path=["Digestive problems", "Diarrhea"],
    ),
    make_chunk(
        "dog-skin",
        "dog",
        "Itching and Skin Disease in Dogs",
        "Itching and skin lesions in dogs commonly reflect allergic skin disease, fleas, or "
        "secondary infection. Hair loss and reddened skin warrant a veterinary examination.",
        categories=["Skin"],
        heading_path=["Dermatology"],
    ),
    make_chunk(
        "dog-limp",
        "dog",
        "Limping in Dogs",
        "Limping and lameness in dogs can follow cruciate ligament injury or patellar "
        "luxation. A dog that will not bear weight on a limb should be seen promptly.",
        categories=["Musculoskeletal"],
        heading_path=["Orthopedics"],
    ),
    make_chunk(
        "dog-seizure",
        "dog",
        "Seizures in Dogs",
        "A seizure in a dog is an emergency when it lasts more than five minutes or repeats. "
        "Owners should note duration and recovery time before seeking emergency veterinary care.",
        categories=["Neurology", "Emergency"],
        heading_path=["Neurology", "Seizures"],
    ),
    make_chunk(
        "dog-chocolate",
        "dog",
        "Chocolate Toxicity in Dogs",
        "Chocolate toxicity in dogs causes tremors, vomiting, and cardiac arrhythmia from "
        "theobromine. Any suspected ingestion requires immediate veterinary attention.",
        categories=["Toxins & poisons", "Emergency"],
        heading_path=["Toxicology", "Chocolate"],
    ),
    make_chunk(
        "dog-dental",
        "dog",
        "Periodontal Disease in Dogs",
        "Periodontal disease in dogs produces bad breath, calculus, and loose teeth. "
        "Owner observations of chewing behaviour guide the timing of dental treatment.",
        categories=["Dental"],
        heading_path=["Dentistry"],
    ),
    make_chunk(
        "dog-weight",
        "dog",
        "Weight Loss in Dogs",
        "Unexplained weight loss in dogs may indicate kidney disease, cancer, or "
        "malabsorption. Progressive weight loss with a normal appetite is a red flag.",
        categories=["Internal medicine"],
        heading_path=["Internal medicine", "Weight loss"],
    ),
]

#: 고양이 문서 6건. 강아지 문서와 **어휘가 상당히 겹치도록** 만든다.
#: 어휘가 겹쳐야 "종이 섞일 수 있는 조건"이 실제로 성립하고, 그럼에도 섞이지 않는지를
#: 확인해야 species 분리 테스트가 의미를 갖는다.
CAT_CHUNKS: list[FakeChunk] = [
    make_chunk(
        "cat-vomit",
        "cat",
        "Vomiting in Cats",
        "Vomiting in cats can reflect hairballs, kidney disease, or hyperthyroidism. "
        "Red flags include repeated vomiting, weight loss, and a cat that will not eat.",
        categories=["Digestive system", "Emergency"],
        heading_path=["Digestive problems", "Vomiting"],
    ),
    make_chunk(
        "cat-urinary",
        "cat",
        "Feline Lower Urinary Tract Disease",
        "Straining to urinate is a urinary obstruction emergency in male cats. "
        "A blocked cat needs immediate veterinary attention within hours.",
        categories=["Urinary", "Emergency"],
        heading_path=["Urology", "FLUTD"],
    ),
    make_chunk(
        "cat-kidney",
        "cat",
        "Chronic Kidney Disease in Cats",
        "Chronic kidney disease is common in older cats and causes increased thirst, "
        "urination, and weight loss. Owner observations of water intake help staging.",
        categories=["Kidney", "Senior care"],
        heading_path=["Nephrology"],
    ),
    make_chunk(
        "cat-lily",
        "cat",
        "Lily Toxicity in Cats",
        "Lily toxicity in cats causes acute kidney injury even from pollen exposure. "
        "Any suspected ingestion requires immediate veterinary attention.",
        categories=["Toxins & poisons", "Emergency"],
        heading_path=["Toxicology", "Lilies"],
    ),
    make_chunk(
        "cat-asthma",
        "cat",
        "Feline Asthma",
        "Cough and respiratory distress in cats often reflect feline asthma. "
        "Open mouth breathing in a cat is always an emergency.",
        categories=["Respiratory system", "Emergency"],
        heading_path=["Respiratory problems", "Asthma"],
    ),
    make_chunk(
        "cat-dental",
        "cat",
        "Dental Disease in Cats",
        "Periodontal disease and tooth resorption in cats cause bad breath and drooling. "
        "Owner observations of chewing behaviour guide dental treatment.",
        categories=["Dental"],
        heading_path=["Dentistry"],
    ),
]

ALL_CHUNKS: list[FakeChunk] = [*DOG_CHUNKS, *CAT_CHUNKS]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_settings(**rag_overrides: Any) -> Settings:
    """오프라인 검색용 `Settings` — 항상 deterministic 백엔드를 쓴다.

    전역 `get_settings()` 싱글턴을 건드리면 테스트 간 간섭이 생기므로,
    각 테스트가 자기 `Settings` 인스턴스를 만들어 주입한다.
    """
    rag = RagSettings(embedding_backend="deterministic", **rag_overrides)
    return Settings(rag=rag)


@pytest.fixture(scope="module")
def embeddings() -> DeterministicEmbeddings:
    """모든 테스트가 공유하는 결정적 임베딩(모델 다운로드 없음)."""
    return DeterministicEmbeddings()


@pytest.fixture(scope="module")
def settings() -> Settings:
    """검색 테스트 공용 설정."""
    return make_settings()


@pytest.fixture(scope="module")
def store(settings: Settings, embeddings: DeterministicEmbeddings) -> VeterinaryVectorStore:
    """dog/cat index 가 모두 올라간 메모리 전용 store(디스크 산출물 없음)."""
    built = VeterinaryVectorStore(settings=settings, embeddings=embeddings)
    built.build_all(ALL_CHUNKS)
    return built


class RecordingStore(VeterinaryVectorStore):
    """`search()` 가 어떤 species 로 호출됐는지 기록하는 store.

    retriever 가 "다른 종 결과를 걸러낸다"가 아니라 "애초에 다른 종 index 를
    조회하지 않는다"를 지키는지 확인하려면 호출 자체를 관찰해야 한다.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.searched_species: list[str] = []

    def search(
        self, query: str, species: Species, k: int = 6, fetch_k: int = 20
    ) -> list[RetrievedEvidence]:
        self.searched_species.append(species)
        return super().search(query, species, k=k, fetch_k=fetch_k)


# ---------------------------------------------------------------------------
# 1. DeterministicEmbeddings (명세 11절)
# ---------------------------------------------------------------------------
class TestDeterministicEmbeddings:
    """오프라인 임베딩이 index 재현성의 전제 조건을 만족하는지 본다."""

    def test_같은_입력은_항상_같은_벡터를_만든다(self) -> None:
        """결정성 검증.

        index 를 저장했다가 다시 로드해 검색해도 같은 결과가 나오려면, 임베딩이
        프로세스/인스턴스와 무관하게 결정적이어야 한다. 파이썬 `hash()` 는 실행마다
        salt 가 달라지므로 그것을 썼다면 여기서 깨진다.
        """
        text = "노령견의 기침과 호흡곤란 경고 신호"
        first = DeterministicEmbeddings().embed_query(text)
        second = DeterministicEmbeddings().embed_query(text)
        assert first == second

    def test_문서_임베딩과_질의_임베딩이_같은_공간에_있다(
        self, embeddings: DeterministicEmbeddings
    ) -> None:
        """같은 문자열이면 embed_documents 와 embed_query 결과가 동일해야 한다.

        두 경로가 다른 벡터를 만들면 코사인 유사도가 의미를 잃고, FAISS 내적 점수를
        "코사인"이라고 부르는 vector store 의 계약이 통째로 거짓이 된다.
        """
        text = "warning signs of cough in an older dog"
        assert embeddings.embed_documents([text])[0] == embeddings.embed_query(text)

    def test_벡터_차원이_384로_고정된다(self, embeddings: DeterministicEmbeddings) -> None:
        """차원 고정 검증 — 저장된 index 와의 호환성 기준이 되는 값이다.

        `DETERMINISTIC_DIM` 이 흔들리면 이미 저장된 테스트 index 가 전부 무효가 된다.
        """
        assert DETERMINISTIC_DIM == 384
        vectors = embeddings.embed_documents(["강아지가 토해요", "cat vomiting"])
        assert all(len(vector) == 384 for vector in vectors)
        assert len(embeddings.embed_query("질의")) == 384

    def test_L2_정규화되어_내적이_코사인_유사도가_된다(
        self, embeddings: DeterministicEmbeddings
    ) -> None:
        """정규화 검증.

        vector store 는 `IndexFlatIP`(내적)를 쓰고 그 점수를 코사인이라고 해석한다.
        벡터 노름이 1이 아니면 그 해석이 깨지고 `min_relevance_score` 같은 임계값
        비교가 전부 틀어진다.
        """
        for text in ["강아지가 기침을 해요", "vomiting in dogs", "a"]:
            vector = embeddings.embed_query(text)
            norm = sum(value * value for value in vector) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_정규화를_끄면_노름이_1이_아니다(self) -> None:
        """`normalize=False` 옵션이 실제로 동작하는지 확인한다.

        정규화가 항상 켜져 있다면 옵션이 죽은 코드라는 뜻이고, 반대로 옵션이
        무시되면 vector store 가 자체 정규화로 덮어쓴다는 사실을 보장할 수 없다.
        """
        raw = DeterministicEmbeddings(normalize=False).embed_query("vomiting in dogs")
        norm = sum(value * value for value in raw) ** 0.5
        assert norm > 1.0

    def test_빈_문자열과_공백_입력에도_안전하다(
        self, embeddings: DeterministicEmbeddings
    ) -> None:
        """빈 입력 방어 검증.

        토큰이 하나도 없으면 0 벡터가 되고, 0 벡터를 정규화하면 0 나눗셈이 난다.
        빈 문서/빈 query 는 실제로 들어오므로 예외 대신 결정적 단위 벡터를 돌려줘야 한다.
        """
        for text in ["", "   ", "\n\t"]:
            vector = embeddings.embed_query(text)
            assert len(vector) == 384
            norm = sum(value * value for value in vector) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-6)
        assert embeddings.embed_documents(["", "  "]) == [
            embeddings.embed_query(""),
            embeddings.embed_query("  "),
        ]

    def test_잘못된_차원은_생성_시점에_거부한다(self) -> None:
        """0 이하 차원은 만들어지자마자 실패해야 한다 — 나중에 faiss 안에서 터지면 원인 추적이 어렵다."""
        with pytest.raises(ValueError):
            DeterministicEmbeddings(dim=0)

    def test_팩토리가_deterministic_백엔드를_돌려준다(self) -> None:
        """`build_embeddings()` 가 설정만 보고 오프라인 백엔드를 고르는지 확인한다.

        이 경로가 깨지면 단위테스트가 실제 bge-m3 를 내려받으려 하면서 CI 가 멈춘다.
        """
        built = build_embeddings(make_settings().rag)
        assert isinstance(built, DeterministicEmbeddings)
        assert embedding_dimension(built) == 384

    def test_지원하지_않는_백엔드는_ValueError(self) -> None:
        """오타 난 백엔드 이름이 조용히 huggingface 로 넘어가면 안 된다."""
        rag = RagSettings(embedding_backend="deterministic")
        rag.embedding_backend = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError):
            build_embeddings(rag)


# ---------------------------------------------------------------------------
# 2~3. build → save → load 왕복과 설정 지문 (명세 11절)
# ---------------------------------------------------------------------------
class TestVectorStorePersistence:
    """index 를 재사용해도 되는지 판정하는 규칙을 검증한다."""

    def test_build_save_load_왕복이_같은_검색_결과를_준다(self, tmp_path: Path) -> None:
        """왕복 검증 + `load()` 가 bool 을 돌려주는지 확인한다.

        노트북은 `if not store.load(): store.build_all(...)` 한 줄로 재생성을 결정한다.
        따라서 반환값이 반드시 bool 이어야 하고, 로드된 index 는 build 직후와 동일한
        검색 결과를 줘야 한다(그렇지 않으면 재사용이 조용히 품질을 떨어뜨린다).
        """
        cfg = make_settings()
        original = VeterinaryVectorStore(settings=cfg, embeddings=DeterministicEmbeddings())
        original.build_all(ALL_CHUNKS)
        original.save(tmp_path)

        assert (tmp_path / "faiss_dog" / "meta.json").exists()
        assert (tmp_path / "faiss_cat" / "records.json").exists()

        restored = VeterinaryVectorStore(settings=cfg, embeddings=DeterministicEmbeddings())
        result = restored.load(tmp_path)

        assert isinstance(result, bool)
        assert result is True
        assert restored.loaded_species == {"dog", "cat"}
        assert restored.index_size("dog") == len(DOG_CHUNKS)
        assert restored.index_size("cat") == len(CAT_CHUNKS)

        query = "vomiting warning signs"
        before = [item.chunk_id for item in original.search(query, "dog", k=5)]
        after = [item.chunk_id for item in restored.search(query, "dog", k=5)]
        assert before == after

    def test_저장된_index_가_없으면_load_는_False(self, tmp_path: Path) -> None:
        """빈 디렉터리에서 load 하면 예외가 아니라 False 여야 한다.

        최초 실행(=index 없음)은 오류가 아니라 "지금 만들어야 한다"는 정상 신호다.
        """
        store = VeterinaryVectorStore(
            settings=make_settings(), embeddings=DeterministicEmbeddings()
        )
        assert store.load(tmp_path / "empty") is False
        assert store.loaded_species == set()

    def test_chunk_size_가_바뀌면_옛_index_를_쓰지_않는다(self, tmp_path: Path) -> None:
        """chunk 파라미터 변경 → `load()=False`.

        chunk_size 가 바뀌면 record 경계 자체가 달라지므로, 옛 index 를 그대로 쓰면
        "설정은 새것, 검색은 옛것"인 상태로 조용히 틀린 근거를 낸다.
        """
        saved = VeterinaryVectorStore(
            settings=make_settings(chunk_size=1000), embeddings=DeterministicEmbeddings()
        )
        saved.build(DOG_CHUNKS, "dog")
        saved.save(tmp_path)

        changed = VeterinaryVectorStore(
            settings=make_settings(chunk_size=500), embeddings=DeterministicEmbeddings()
        )
        assert changed.load(tmp_path) is False
        assert changed.loaded_species == set()

    def test_embedding_model_이_바뀌면_옛_index_를_쓰지_않는다(self, tmp_path: Path) -> None:
        """임베딩 모델 변경 → `load()=False`.

        모델이 바뀌면 벡터 공간이 완전히 달라진다. 차원이 우연히 같아도 옛 벡터는
        무의미하므로 차원 검사만으로는 부족하고 지문에 모델명이 들어가야 한다.
        """
        saved = VeterinaryVectorStore(
            settings=make_settings(embedding_model="BAAI/bge-m3"),
            embeddings=DeterministicEmbeddings(),
        )
        saved.build(DOG_CHUNKS, "dog")
        saved.save(tmp_path)

        changed = VeterinaryVectorStore(
            settings=make_settings(embedding_model="intfloat/multilingual-e5-base"),
            embeddings=DeterministicEmbeddings(),
        )
        assert changed.load(tmp_path) is False

    def test_임베딩_차원이_바뀌면_옛_index_를_쓰지_않는다(self, tmp_path: Path) -> None:
        """임베딩 차원 변경 → `load()=False`.

        설정 문자열은 그대로인데 주입된 임베딩의 차원만 달라지는 경우다. 지문의
        클래스 이름만으로는 못 걸러내고, 막지 못하면 faiss 검색 시점에 뒤늦게 터진다.
        """
        cfg = make_settings()
        saved = VeterinaryVectorStore(settings=cfg, embeddings=DeterministicEmbeddings(dim=384))
        saved.build(DOG_CHUNKS, "dog")
        saved.save(tmp_path)

        changed = VeterinaryVectorStore(
            settings=cfg, embeddings=DeterministicEmbeddings(dim=128)
        )
        assert changed.load(tmp_path) is False

    def test_정규화_설정이_바뀌면_옛_index_를_쓰지_않는다(self, tmp_path: Path) -> None:
        """`embedding_normalize` 변경 → `load()=False` (지문에 포함되는지 확인)."""
        saved = VeterinaryVectorStore(
            settings=make_settings(embedding_normalize=True),
            embeddings=DeterministicEmbeddings(),
        )
        saved.build(CAT_CHUNKS, "cat")
        saved.save(tmp_path)

        changed = VeterinaryVectorStore(
            settings=make_settings(embedding_normalize=False),
            embeddings=DeterministicEmbeddings(normalize=False),
        )
        assert changed.load(tmp_path) is False

    def test_설정이_같으면_index_를_그대로_재사용한다(self, tmp_path: Path) -> None:
        """지문이 동일하면 반드시 재사용돼야 한다 — 과잉 무효화도 버그다.

        매번 False 를 돌려주는 구현은 "안전"해 보이지만 Colab 에서 매 실행마다
        283문서를 재임베딩하게 만들어 명세 11절의 재사용 요구를 위반한다.
        """
        cfg = make_settings()
        saved = VeterinaryVectorStore(settings=cfg, embeddings=DeterministicEmbeddings())
        saved.build(DOG_CHUNKS, "dog")
        saved.save(tmp_path)

        reused = VeterinaryVectorStore(
            settings=make_settings(), embeddings=DeterministicEmbeddings()
        )
        assert reused.load(tmp_path) is True
        assert reused.loaded_species == {"dog"}


# ---------------------------------------------------------------------------
# 4~6. 종 분리와 metadata 보존 (명세 11·13·17절)
# ---------------------------------------------------------------------------
class TestSpeciesIsolation:
    """강아지/고양이 문서를 절대 섞지 않는다는 요구를 여러 층위에서 확인한다."""

    def test_같은_query_라도_dog_와_cat_결과의_교집합이_없다(
        self, store: VeterinaryVectorStore
    ) -> None:
        """species 분리의 핵심 검증.

        dog/cat 문서는 "vomiting", "dental", "emergency" 같은 어휘를 공유하므로
        index 가 하나였다면 반드시 섞였을 query 를 쓴다. 그럼에도 chunk_id 교집합이
        공집합이면 분리가 index 구조 수준에서 보장된다는 뜻이다.
        """
        for query in ["vomiting red flags", "dental disease bad breath", "emergency"]:
            dog_ids = {item.chunk_id for item in store.search(query, "dog", k=6)}
            cat_ids = {item.chunk_id for item in store.search(query, "cat", k=6)}
            assert dog_ids, f"dog 검색 결과가 비었습니다: {query}"
            assert cat_ids, f"cat 검색 결과가 비었습니다: {query}"
            assert dog_ids & cat_ids == set()

    def test_검색_결과의_species_는_항상_요청한_종이다(
        self, store: VeterinaryVectorStore
    ) -> None:
        """`RetrievedEvidence.species` 는 metadata 가 아니라 검색한 index 를 따라야 한다."""
        for species in ("dog", "cat"):
            results = store.search("vomiting weight loss", species, k=6)  # type: ignore[arg-type]
            assert results
            assert all(item.species == species for item in results)

    def test_고양이_질문_결과에_강아지_문서가_섞이지_않는다(
        self, store: VeterinaryVectorStore, settings: Settings
    ) -> None:
        """retrieve() 레벨에서도 종 분리가 유지되는지 확인한다(명세 17절).

        store.search 단위로는 분리돼 있어도 retriever 가 ko/en 결과를 합치면서
        다른 종 batch 를 끼워 넣으면 오염이 발생한다. 실제 사용 경로로 다시 본다.
        """
        query = build_rag_query(
            "고양이가 토하고 물을 많이 마셔요",
            {"species": "고양이", "age_years": 12},
            [],
            [],
        )
        assert query.species == "cat"

        evidence = retrieve(store, query, settings=settings)
        assert evidence
        dog_document_ids = {chunk.document_id for chunk in DOG_CHUNKS}
        assert all(item.species == "cat" for item in evidence)
        assert not ({item.document_id for item in evidence} & dog_document_ids)

    def test_build_는_다른_종_chunk_를_스스로_걸러낸다(self, settings: Settings) -> None:
        """호출자가 전체 chunk 를 통째로 넘겨도 index 에는 해당 종만 담겨야 한다.

        `build_all()` 이 아니라 `build(전체, 'dog')` 로 실수하는 경우가 실제로 흔하다.
        이 2차 방어선이 없으면 그 순간 dog index 에 cat 문서가 들어간다.
        """
        store = VeterinaryVectorStore(settings=settings, embeddings=DeterministicEmbeddings())
        store.build(ALL_CHUNKS, "dog")
        assert store.index_size("dog") == len(DOG_CHUNKS)
        assert store.loaded_species == {"dog"}

    def test_해당_종_chunk_가_하나도_없으면_명확히_실패한다(self, settings: Settings) -> None:
        """빈 index 를 조용히 만들면 나중에 "검색 결과 0건"의 원인을 못 찾는다."""
        store = VeterinaryVectorStore(settings=settings, embeddings=DeterministicEmbeddings())
        with pytest.raises(ValueError):
            store.build(DOG_CHUNKS, "cat")

    def test_지원하지_않는_species_는_거부한다(self, settings: Settings) -> None:
        """dog/cat 외의 종은 index 경로 자체가 없으므로 build 단계에서 막는다."""
        store = VeterinaryVectorStore(settings=settings, embeddings=DeterministicEmbeddings())
        with pytest.raises(ValueError):
            store.build(DOG_CHUNKS, "rabbit")  # type: ignore[arg-type]


class TestMetadataPreservation:
    """검색 결과가 "출처를 밝힐 수 있는" 상태로 오는지 확인한다(명세 11·13절)."""

    def test_RetrievedEvidence_가_metadata_를_모두_보존한다(
        self, store: VeterinaryVectorStore
    ) -> None:
        """title / source_url / heading_path / categories 보존 검증.

        답변에서 근거를 인용하려면 이 네 값이 반드시 살아 있어야 한다. chunk →
        record(json) → faiss 저장 → RetrievedEvidence 로 넘어가는 사이에 어느 한
        단계라도 키를 흘리면 출처 없는 근거가 되어 명세 위반이다.
        """
        results = store.search("chocolate toxicity theobromine tremors", "dog", k=6)
        matched = {item.chunk_id: item for item in results}
        assert "dog-chocolate#0" in matched

        evidence = matched["dog-chocolate#0"]
        source = next(chunk for chunk in DOG_CHUNKS if chunk.chunk_id == "dog-chocolate#0")

        assert evidence.title == source.metadata["title"]
        assert evidence.source == source.metadata["source"]
        assert evidence.source_url == source.metadata["source_url"]
        assert evidence.heading_path == source.metadata["heading_path"]
        assert evidence.categories == source.metadata["categories"]
        assert evidence.document_id == source.document_id
        assert evidence.text == source.text

    def test_metadata_는_저장과_로드를_거쳐도_유지된다(self, tmp_path: Path) -> None:
        """디스크 왕복 후에도 heading_path/categories 가 살아 있는지 본다.

        records.json 직렬화에서 리스트형 metadata 가 깨지는 사고를 잡기 위한 테스트다.
        """
        cfg = make_settings()
        saved = VeterinaryVectorStore(settings=cfg, embeddings=DeterministicEmbeddings())
        saved.build_all(ALL_CHUNKS)
        saved.save(tmp_path)

        restored = VeterinaryVectorStore(settings=cfg, embeddings=DeterministicEmbeddings())
        assert restored.load(tmp_path) is True

        results = restored.search("lily toxicity acute kidney injury pollen", "cat", k=6)
        evidence = next(item for item in results if item.chunk_id == "cat-lily#0")
        assert evidence.title == "Lily Toxicity in Cats"
        assert evidence.heading_path == ["Toxicology", "Lilies"]
        assert evidence.categories == ["Toxins & poisons", "Emergency"]
        assert evidence.source_url.endswith("cat-lily")

    def test_score_는_0과_1_사이의_유사도다(self, store: VeterinaryVectorStore) -> None:
        """score 계약 검증 — 거리(작을수록 좋음)가 아니라 유사도(클수록 좋음)다.

        이 값이 `min_relevance_score` 와 직접 비교되므로 범위와 방향이 뒤집히면
        충분성 판단(명세 14절) 전체가 반대로 동작한다.
        """
        results = store.search("vomiting in dogs red flags lethargy", "dog", k=6)
        assert results
        assert all(0.0 <= (item.score or 0.0) <= 1.0 for item in results)
        # 질의와 어휘가 가장 많이 겹치는 문서가 1위여야 한다.
        assert results[0].chunk_id == "dog-vomit#0"


# ---------------------------------------------------------------------------
# 7~9. Query Builder (명세 12절)
# ---------------------------------------------------------------------------
class TestQueryBuilderRuleBased:
    """LLM 없이(=API 키 없음) 동작하는 규칙 기반 경로를 검증한다."""

    def test_한국어와_영어_query_를_모두_만든다(self) -> None:
        """ko/en 동시 생성 검증.

        retriever 는 두 query 를 모두 던진다. 한쪽이 비면 검색이 절반만 돌아가고,
        영어가 비면 영어 코퍼스와의 직접 매칭을 통째로 잃는다.
        """
        query = build_rag_query(
            "우리 강아지가 기침을 자주 해요",
            {"species": "강아지", "breed": "푸들", "age_years": 3},
            [],
            [],
            llm=None,
        )
        assert query.primary_query_ko.strip()
        assert query.primary_query_en.strip()
        assert isinstance(query, RagQuery)

    def test_영어_query_는_입력_원문과_다르고_한글이_없다(self) -> None:
        """"사용자 문장을 그대로 쓰지 않는다"는 명세 12절 요구 검증.

        Cornell 문서는 전부 영어라, 한국어 원문을 그대로 던지면 임베딩 공간에서
        겉돈다. 영어 query 는 원문과 달라야 하고 한글이 섞여 있어서도 안 된다.
        """
        message = "우리 강아지가 기침을 자주 해요"
        query = build_rag_query(
            message,
            {"species": "강아지", "breed": "푸들", "age_years": 3},
            [],
            [],
            llm=None,
        )
        assert query.primary_query_en != message
        assert not _has_hangul(query.primary_query_en)
        assert "cough" in query.primary_query_en.lower()
        assert "기침" in query.primary_query_ko

    def test_species_는_pet_profile_에서_결정된다(self) -> None:
        """species 결정 경로 검증.

        species 는 "어느 index 를 검색할 것인가"를 정하는 값이라 사용자 문장이 아니라
        PET DB 를 따라야 한다. 문장에 다른 종 단어가 들어 있어도 흔들리면 안 된다.
        """
        assert build_rag_query("토해요", {"species": "고양이"}, [], [], llm=None).species == "cat"
        assert build_rag_query("토해요", {"species": "cat"}, [], [], llm=None).species == "cat"
        assert build_rag_query("토해요", {"species": "강아지"}, [], [], llm=None).species == "dog"
        assert build_rag_query("토해요", {"species": "dog"}, [], [], llm=None).species == "dog"
        # 값이 없거나 알 수 없으면 검색을 실패시키지 않고 기본값(dog)으로 진행한다.
        assert build_rag_query("토해요", {}, [], [], llm=None).species == "dog"
        assert build_rag_query("토해요", {"species": "앵무새"}, [], [], llm=None).species == "dog"

    def test_7살_이상이면_노령_표현이_들어간다(self) -> None:
        """노령 반영 검증 — 같은 증상이라도 노령이면 감별 질환이 달라진다.

        Cornell 문서가 실제로 "older dog / senior cat" 절을 따로 두므로, 나이
        수식어가 검색어에 반영되지 않으면 노령 특이 문서를 놓친다.
        """
        senior = build_rag_query(
            "기침을 해요",
            {"species": "강아지", "age_years": SENIOR_AGE_YEARS + 2},
            [],
            [],
            llm=None,
        )
        assert "노령" in senior.primary_query_ko
        assert "older" in senior.primary_query_en.lower()

        young = build_rag_query(
            "기침을 해요", {"species": "강아지", "age_years": 3}, [], [], llm=None
        )
        assert "노령" not in young.primary_query_ko
        assert "older" not in young.primary_query_en.lower()

    def test_고양이_노령도_한국어_영어_모두_반영된다(self) -> None:
        """종에 따라 노령 표현이 달라지는지 확인한다(노령견 vs 노령묘)."""
        query = build_rag_query(
            "밥을 잘 안 먹어요", {"species": "고양이", "age_years": 14}, [], [], llm=None
        )
        assert "노령묘" in query.primary_query_ko
        assert "older cat" in query.primary_query_en.lower()

    def test_기존질환이_query_에_반영된다(self) -> None:
        """PET DB 의 기존질환 반영 검증.

        "기침"만으로는 8살 승모판질환 푸들과 1살 건강한 강아지를 구분할 수 없다.
        질환이 query 에 실려야 심장성 기침 문서 쪽으로 검색이 기운다.
        """
        query = build_rag_query(
            "기침을 해요",
            {"species": "강아지", "age_years": 9, "diseases": ["승모판 질환"]},
            [],
            [],
            llm=None,
        )
        assert "심장질환" in query.primary_query_ko
        assert "heart disease" in query.primary_query_en.lower()
        assert not _has_hangul(query.primary_query_en)

    def test_진단서의_질환도_보충된다(self) -> None:
        """진단서 DB(우선순위 3)가 PET DB 를 보충하는지 확인한다."""
        query = build_rag_query(
            "물을 많이 마셔요",
            {"species": "고양이", "age_years": 13},
            [{"diagnosis": "chronic kidney disease"}],
            [],
            llm=None,
        )
        assert "신장질환" in query.primary_query_ko
        assert "kidney disease" in query.primary_query_en.lower()

    def test_현재_입력의_증상이_과거_기록보다_앞에_온다(self) -> None:
        """명세 12절 context 우선순위 검증.

        과거 일기장의 증상이 지금 호소를 밀어내면 "지금 무슨 일이 벌어지는가"를
        놓친다. 현재 입력 증상이 반드시 먼저 등장해야 한다.
        """
        query = build_rag_query(
            "오늘은 다리를 절어요",
            {"species": "강아지"},
            [],
            [{"symptoms": "기침"}],
            llm=None,
        )
        ko = query.primary_query_ko
        assert "절뚝거림" in ko
        assert "기침" in ko
        assert ko.index("절뚝거림") < ko.index("기침")

    def test_증상_사전에_없으면_사용자_문장을_버리지_않는다(self) -> None:
        """증상 매칭이 하나도 안 돼도 정보가 사라지면 안 된다.

        빈 query 를 만들면 검색이 통째로 무의미해지므로, 원문을 잘라서라도 쓴다.
        """
        query = build_rag_query(
            "산책 후에 자꾸 두리번거려요", {"species": "강아지"}, [], [], llm=None
        )
        assert "두리번" in query.primary_query_ko
        assert query.primary_query_en.strip()
        assert not _has_hangul(query.primary_query_en)

    def test_required_topics_가_항상_채워진다(self) -> None:
        """충분성 판단(명세 14절)의 coverage 기준이 되는 값이라 비어 있으면 안 된다."""
        query = build_rag_query("기침을 해요", {"species": "강아지"}, [], [], llm=None)
        assert query.required_topics
        assert len(query.required_topics) == len(set(query.required_topics))


class TestQueryBuilderEmergency:
    """응급 힌트 판정 — 낮은 쪽으로 틀리면 안 되는 값이다."""

    @pytest.mark.parametrize(
        "message",
        [
            "강아지가 숨을 못 쉬어요",
            "우리 아이가 쓰러졌어요",
            "경련을 일으켜요",
            "응급인가요? 지금 축 처져 있어요",
            "초콜릿을 먹어버렸어요",
            "고양이가 소변을 못 봐요",
        ],
    )
    def test_응급_표현이면_emergency_hint_가_True(self, message: str) -> None:
        """응급 신호 탐지 검증.

        응급을 놓치면(False negative) 보호자가 병원 방문을 늦춘다. 응급 증상 사전과
        응급 문구 목록 두 경로 모두가 동작해야 한다.
        """
        query = build_rag_query(message, {"species": "강아지"}, [], [], llm=None)
        assert query.emergency_hint is True

    @pytest.mark.parametrize(
        "message",
        ["밥을 조금 남겼어요", "귀를 자주 긁어요", "이빨에 치석이 있어요"],
    )
    def test_일반_증상은_emergency_hint_가_False(self, message: str) -> None:
        """반대 방향 검증 — 모든 질문을 응급으로 표시하면 응급 신호가 무의미해진다."""
        query = build_rag_query(message, {"species": "강아지"}, [], [], llm=None)
        assert query.emergency_hint is False

    def test_응급이면_required_topics_가_확장된다(self) -> None:
        """응급일 때 "언제 즉시 병원에 가야 하는가" 토픽이 추가되는지 확인한다."""
        urgent = build_rag_query("숨을 못 쉬어요", {"species": "강아지"}, [], [], llm=None)
        calm = build_rag_query("귀를 긁어요", {"species": "강아지"}, [], [], llm=None)
        assert set(calm.required_topics) < set(urgent.required_topics)

    def test_과거_기록의_응급_이력은_현재를_응급으로_만들지_않는다(self) -> None:
        """`emergency_hint` 는 **현재 입력**으로만 판정한다(명세 12절).

        과거 진단서에 경련 이력이 있다고 오늘의 가벼운 질문까지 응급으로 올리면
        오탐이 쌓이고 응급 신호의 신뢰도가 무너진다.
        """
        query = build_rag_query(
            "요즘 귀를 자주 긁어요",
            {"species": "강아지"},
            [{"diagnosis": "특발성 뇌전증으로 경련 이력 있음"}],
            [],
            llm=None,
        )
        assert query.emergency_hint is False


# --- LLM stub -----------------------------------------------------------------
class _StubStructuredRunnable:
    """`with_structured_output()` 이 돌려주는 runnable 흉내."""

    def __init__(self, owner: "StubLLM") -> None:
        self._owner = owner

    def invoke(self, messages: Any) -> Any:
        self._owner.invoked_messages.append(messages)
        if isinstance(self._owner.result, Exception):
            raise self._owner.result
        return self._owner.result


class StubLLM:
    """네트워크 없이 structured output 경로만 재현하는 LLM mock.

    `with_structured_output` 에 어떤 스키마가 넘어왔는지 기록해, query_builder 가
    자유 텍스트 파싱이 아니라 structured output 을 쓰는지 직접 확인할 수 있게 한다.
    """

    def __init__(self, result: Any) -> None:
        self.result = result
        self.structured_schemas: list[type] = []
        self.invoked_messages: list[Any] = []

    def with_structured_output(self, schema: type) -> _StubStructuredRunnable:
        self.structured_schemas.append(schema)
        return _StubStructuredRunnable(self)


class TestQueryBuilderWithLLM:
    """LLM 은 품질 향상 옵션일 뿐, 실패해도 파이프라인을 멈추지 않는다."""

    def test_LLM_주입_시_structured_output_을_사용한다(self) -> None:
        """structured output 사용 검증.

        자유 텍스트를 받아 직접 파싱하면 스키마 위반이 런타임 오류로 번진다.
        `RagQuery` 스키마로 `with_structured_output` 을 걸었는지 확인한다.
        """
        improved = RagQuery(
            primary_query_ko="심장질환이 있는 노령견의 야간 기침 경고 신호",
            primary_query_en="nocturnal cough warning signs in an older dog with mitral valve disease",
            required_topics=["cardiac cough"],
            species="dog",
            emergency_hint=False,
        )
        llm = StubLLM(improved)

        query = build_rag_query(
            "밤에 기침을 해요",
            {"species": "강아지", "age_years": 9, "diseases": ["승모판 질환"]},
            [],
            [],
            llm=llm,
        )

        assert llm.structured_schemas == [RagQuery]
        assert llm.invoked_messages, "LLM 이 실제로 호출되지 않았습니다."
        assert query.primary_query_ko == improved.primary_query_ko
        assert query.primary_query_en == improved.primary_query_en
        # 규칙 기반 토픽은 유지하면서 LLM 토픽을 더한다(근거 커버리지 기준 손실 방지).
        assert "cardiac cough" in query.required_topics
        assert "red flags" in query.required_topics

    def test_LLM_이_예외를_던지면_규칙_기반으로_폴백한다(self) -> None:
        """LLM 실패 시 폴백 검증.

        타임아웃·rate limit·스키마 위반은 운영에서 반드시 일어난다. 그때 사용자
        답변 전체가 실패하면 안 되고, 규칙 기반 결과와 정확히 같아야 한다.
        """
        profile = {"species": "고양이", "age_years": 12, "diseases": ["만성 신부전"]}
        message = "물을 많이 마시고 토해요"

        expected = build_rag_query_rule_based(message, profile, [], [])
        llm = StubLLM(RuntimeError("rate limit exceeded"))

        query = build_rag_query(message, profile, [], [], llm=llm)

        assert llm.invoked_messages, "예외 경로에서도 LLM 호출은 시도돼야 합니다."
        assert query.model_dump() == expected.model_dump()

    def test_LLM_이_빈_query_를_주면_규칙_기반으로_폴백한다(self) -> None:
        """빈 문자열은 스키마상 유효하지만 검색으로는 무의미하므로 걸러야 한다."""
        profile = {"species": "강아지"}
        expected = build_rag_query_rule_based("기침을 해요", profile, [], [])
        llm = StubLLM(RagQuery(primary_query_ko="  ", primary_query_en="", species="dog"))

        query = build_rag_query("기침을 해요", profile, [], [], llm=llm)
        assert query.model_dump() == expected.model_dump()

    def test_LLM_은_species_와_응급도를_낮출_수_없다(self) -> None:
        """안전 불변식 검증.

        species 를 LLM 이 바꾸면 고양이 질문에 강아지 문서가 붙는 사고가 나고,
        응급도를 내리면 위험을 축소 보고하게 된다. 둘 다 규칙 기반이 이겨야 한다.
        """
        llm = StubLLM(
            RagQuery(
                primary_query_ko="고양이의 가벼운 호흡 변화",
                primary_query_en="mild breathing change in a cat",
                required_topics=[],
                species="dog",  # 잘못된 종으로 덮어쓰기 시도
                emergency_hint=False,  # 응급 해제 시도
            )
        )

        query = build_rag_query(
            "고양이가 숨을 못 쉬어요", {"species": "고양이"}, [], [], llm=llm
        )

        assert query.species == "cat"
        assert query.emergency_hint is True


# ---------------------------------------------------------------------------
# 10~12. Retriever (명세 13절)
# ---------------------------------------------------------------------------
class TestRetriever:
    """ko/en 병합·중복 제거·상한·species 고정·경계 조건을 검증한다."""

    def test_ko_en_결과를_합쳐_중복_없이_돌려준다(
        self, store: VeterinaryVectorStore, settings: Settings
    ) -> None:
        """병합 + 중복 제거 검증.

        같은 chunk 가 ko/en 양쪽에 걸리는 것은 정상이자 관련도가 높다는 신호다.
        하지만 같은 근거를 두 번 인용하면 프롬프트만 낭비되므로 하나만 남아야 한다.
        """
        query = build_rag_query(
            "강아지가 토하고 기운이 없어요", {"species": "강아지"}, [], [], llm=None
        )
        evidence = retrieve(store, query, settings=settings)

        assert evidence
        chunk_ids = [item.chunk_id for item in evidence]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_결과_개수가_final_evidence_max_이하다(
        self, store: VeterinaryVectorStore
    ) -> None:
        """상한 검증 — 명세 13절 "최종 상위 4~8개".

        index(dog 10건)가 상한보다 크므로 절단이 실제로 일어나는 조건이다.
        """
        query = build_rag_query(
            "강아지가 토해요", {"species": "강아지"}, [], [], llm=None
        )
        for configured, expected in ((4, 4), (8, 8), (100, MAX_FINAL_EVIDENCE), (1, MIN_FINAL_EVIDENCE)):
            cfg = make_settings(final_evidence_max=configured)
            assert resolve_final_evidence_limit(cfg) == expected
            evidence = retrieve(store, query, settings=cfg)
            assert len(evidence) <= expected

    def test_점수_내림차순으로_정렬된다(
        self, store: VeterinaryVectorStore, settings: Settings
    ) -> None:
        """정렬 방향 검증 — score 는 거리가 아니라 유사도라 클수록 앞이다."""
        query = build_rag_query(
            "강아지가 토해요", {"species": "강아지"}, [], [], llm=None
        )
        scores = [item.score or 0.0 for item in retrieve(store, query, settings=settings)]
        assert scores == sorted(scores, reverse=True)

    def test_중복_chunk_는_더_높은_점수를_남긴다(self) -> None:
        """중복 제거 규칙 검증(명세 13절).

        낮은 점수를 남기면 그 근거가 순위에서 부당하게 밀리고, 충분성 판단의
        임계값 비교도 실제보다 박하게 나온다.
        """
        low = _evidence("c1", score=0.31)
        high = _evidence("c1", score=0.87)
        other = _evidence("c2", score=0.55)

        merged = deduplicate_evidence([[low, other], [high]])

        assert [item.chunk_id for item in merged] == ["c1", "c2"]
        assert merged[0].score == pytest.approx(0.87)

    def test_다른_종_evidence_는_병합_단계에서_버려진다(self) -> None:
        """2차 방어선 검증 — metadata 가 오염돼도 최종 근거에는 못 들어온다."""
        merged = deduplicate_evidence(
            [[_evidence("dog-1", score=0.9, species="dog"), _evidence("cat-1", score=0.8, species="cat")]],
            species="cat",
        )
        assert [item.chunk_id for item in merged] == ["cat-1"]

    def test_chunk_id_가_비어도_근거를_버리지_않는다(self) -> None:
        """빈 chunk_id 는 중복 판정이 불가능하지만, 근거를 잃는 것이 더 나쁘다."""
        merged = deduplicate_evidence([[_evidence("", score=0.7), _evidence("", score=0.6)]])
        assert len(merged) == 2

    def test_query_species_이외의_index_는_조회하지_않는다(
        self, settings: Settings
    ) -> None:
        """species 고정 검증(명세 13절).

        "다른 종 결과를 나중에 걸러낸다"로는 부족하다. 애초에 다른 종 index 를
        조회하지 않아야 오염 가능성 자체가 사라지고 검색 비용도 절반이다.
        """
        store = RecordingStore(settings=settings, embeddings=DeterministicEmbeddings())
        store.build_all(ALL_CHUNKS)
        assert store.loaded_species == {"dog", "cat"}

        query = build_rag_query(
            "고양이가 소변을 못 봐요", {"species": "고양이"}, [], [], llm=None
        )
        evidence = retrieve(store, query, settings=settings)

        assert set(store.searched_species) == {"cat"}
        assert len(store.searched_species) == 2  # ko / en 두 번
        assert evidence
        assert all(item.species == "cat" for item in evidence)


class TestRetrievalEdgeCases:
    """경계 조건에서 예외가 아니라 빈 결과로 흡수되는지 확인한다."""

    def test_k_가_index_크기보다_커도_안전하다(self, store: VeterinaryVectorStore) -> None:
        """k > index 크기 검증.

        faiss 는 부족한 자리를 -1 로 채워 돌려준다. 그 -1 을 record 인덱스로 쓰면
        IndexError 가 나므로 반드시 걸러져야 한다.
        """
        results = store.search("vomiting", "cat", k=500, fetch_k=1000)
        assert 0 < len(results) <= len(CAT_CHUNKS)
        assert all(item.chunk_id for item in results)

    def test_빈_query_는_빈_결과를_돌려준다(self, store: VeterinaryVectorStore) -> None:
        """빈 query 검색은 오류가 아니라 "근거 없음"이다(→ 웹 fallback 정상 경로)."""
        assert store.search("", "dog") == []
        assert store.search("   ", "dog") == []

    def test_로드되지_않은_species_는_빈_결과를_돌려준다(self, settings: Settings) -> None:
        """미로드 index 검증 — 예외를 던지면 파이프라인 전체가 죽는다."""
        store = VeterinaryVectorStore(settings=settings, embeddings=DeterministicEmbeddings())
        store.build(DOG_CHUNKS, "dog")
        assert store.loaded_species == {"dog"}
        assert store.search("vomiting", "cat") == []
        assert store.index_size("cat") == 0

    def test_미로드_species_로_retrieve_해도_빈_리스트다(self, settings: Settings) -> None:
        """retrieve() 레벨에서도 같은 계약이 유지되는지 본다."""
        store = VeterinaryVectorStore(settings=settings, embeddings=DeterministicEmbeddings())
        store.build(DOG_CHUNKS, "dog")

        query = build_rag_query("토해요", {"species": "고양이"}, [], [], llm=None)
        assert retrieve(store, query, settings=settings) == []

    def test_ko_en_query_가_모두_비면_빈_리스트다(
        self, store: VeterinaryVectorStore, settings: Settings
    ) -> None:
        """빈 RagQuery 방어 — 검색어가 없으면 조용히 빈 결과로 끝나야 한다."""
        empty = RagQuery(primary_query_ko="  ", primary_query_en="", species="dog")
        assert retrieve(store, empty, settings=settings) == []

    def test_MMR_을_꺼도_검색이_동작한다(self, embeddings: DeterministicEmbeddings) -> None:
        """`use_mmr=False` 경로 검증 — 설정 분기 중 한쪽만 테스트되면 나머지가 썩는다."""
        cfg = make_settings(use_mmr=False)
        store = VeterinaryVectorStore(settings=cfg, embeddings=embeddings)
        store.build(DOG_CHUNKS, "dog")

        results = store.search("vomiting in dogs red flags", "dog", k=3)
        assert len(results) == 3
        assert results[0].chunk_id == "dog-vomit#0"

    def test_stats_가_index_상태를_보고한다(self, store: VeterinaryVectorStore) -> None:
        """노트북 진단용 요약이 실제 index 상태와 일치하는지 확인한다."""
        stats = store.stats()
        assert stats["loaded_species"] == ["cat", "dog"]
        assert stats["counts"] == {"dog": len(DOG_CHUNKS), "cat": len(CAT_CHUNKS)}
        assert stats["dimensions"] == {"dog": 384, "cat": 384}


def _evidence(
    chunk_id: str, score: float, species: Species = "dog", document_id: str = "doc"
) -> RetrievedEvidence:
    """중복 제거 테스트용 최소 `RetrievedEvidence` 팩토리."""
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title="t",
        text="body",
        species=species,
        source="src",
        source_url="https://www.vet.cornell.edu/x",
        score=score,
    )

from __future__ import annotations

from dataclasses import dataclass

from petcare_agent.rag.cornell import CornellRAGAdapter


@dataclass(frozen=True)
class FakeCornellChunk:
    chunk_id: str = "cornell_dog_chocolate_001"
    document_id: str = "cornell_dog_chocolate"
    title: str = "Chocolate toxicity"
    section_path: list[str] | None = None
    species: list[str] | None = None
    canonical_url: str = "https://www.vet.cornell.edu/chocolate-toxicity"
    content: str = "Chocolate can be dangerous for dogs."
    distance: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_path", self.section_path or [self.title])
        object.__setattr__(self, "species", self.species or ["dog"])

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


def test_cornell_adapter_maps_team_rag_chunk_to_project_chunk() -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_retriever(question: str, species: str, top_k: int):
        calls.append((question, species, top_k))
        return [FakeCornellChunk()]

    adapter = CornellRAGAdapter(retriever=fake_retriever)

    result = adapter.retrieve("dog ate chocolate", {"species": "dog"}, top_k=3)

    assert calls == [("dog ate chocolate", "dog", 3)]
    assert len(result) == 1
    chunk = result[0]
    assert chunk.chunk_id == "cornell_dog_chocolate_001"
    assert chunk.source_id == "cornell_dog_chocolate"
    assert chunk.title == "Chocolate toxicity"
    assert chunk.text == "Chocolate can be dangerous for dogs."
    assert chunk.score == 0.8
    assert chunk.metadata["provider"] == "cornell"
    assert chunk.metadata["canonical_url"].startswith("https://www.vet.cornell.edu/")
    assert chunk.metadata["section_path"] == ["Chocolate toxicity"]


def test_cornell_adapter_rejects_unknown_species_before_retriever_call() -> None:
    calls = 0

    def fake_retriever(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [FakeCornellChunk()]

    adapter = CornellRAGAdapter(retriever=fake_retriever)

    assert adapter.retrieve("rabbit question", {"species": "rabbit"}) == []
    assert calls == 0


def test_cornell_adapter_filters_wrong_species_results() -> None:
    adapter = CornellRAGAdapter(
        retriever=lambda *_args, **_kwargs: [FakeCornellChunk(species=["cat"])],
    )

    assert adapter.retrieve("dog question", {"species": "dog"}) == []

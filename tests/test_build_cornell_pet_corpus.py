from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "build_cornell_pet_corpus.py"
SPEC = importlib.util.spec_from_file_location("cornell_builder", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class WordCounter:
    """Small deterministic counter used to test chunk behavior without a package."""

    def count(self, text: str) -> int:
        return len(text.split())


def make_document(
    *,
    document_id: str = "cornell_cat_test",
    title: str = "Test",
    body: str = "# Test\n\nUseful medical information.",
    original_hash: str = "same",
    filename: str = "test.md",
) -> object:
    return builder.SourceDocument(
        source_path=Path(filename),
        raw_id=document_id.replace("_", ":", 2),
        document_id=document_id,
        title=title,
        species="cat",
        source_center="Cornell Feline Health Center",
        categories=("Test",),
        last_updated=None,
        language="en",
        medical_domain="feline_health",
        original_content_hash=original_hash,
        body=body,
        canonical_url="https://www.vet.cornell.edu/test",
        content_hash=builder.hash_content(body),
    )


class FrontmatterTests(unittest.TestCase):
    def test_parses_frontmatter_and_categories(self) -> None:
        text = """---
id: "cornell:cat:test"
title: "Test"
species: "cat"
source: "Cornell Feline Health Center"
categories: ["One", "Two"]
last_updated: ""
language: "en"
medical_domain: "feline_health"
content_hash: "abc"
---
# Test

Body.
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.md"
            path.write_text(text, encoding="utf-8")
            metadata, body = builder.parse_frontmatter(path)
        self.assertEqual(metadata["categories"], ["One", "Two"])
        self.assertEqual(metadata["last_updated"], "")
        self.assertEqual(body, "# Test\n\nBody.")

    def test_normalizes_id(self) -> None:
        self.assertEqual(
            builder.normalize_document_id("cornell:dog:vomiting", "dog", "Vomiting"),
            "cornell_dog_vomiting",
        )


class CleaningTests(unittest.TestCase):
    def test_removes_suggested_articles_but_keeps_medical_intro(self) -> None:
        body = """# Vomiting

### Suggested Articles

Diarrhea

Video: Caring for Your Cat

Cats that vomit repeatedly should be evaluated by a veterinarian.
"""
        cleaned, removed = builder.remove_suggested_articles(
            body, {builder.normalized_title("Diarrhea")}
        )
        self.assertEqual(removed, 3)
        self.assertNotIn("Suggested Articles", cleaned)
        self.assertNotIn("Video:", cleaned)
        self.assertIn("Cats that vomit repeatedly", cleaned)

    def test_removes_suggested_articles_and_resources_variant(self) -> None:
        body = """# Rabies

### Suggested Articles and Resources

Feline Vaccines: Benefits and Risks

Centers for Disease Control - Rabies

Q: Why does an indoor cat need a rabies vaccine?
"""
        cleaned, removed = builder.remove_suggested_articles(
            body, {builder.normalized_title("Feline Vaccines: Benefits and Risks")}
        )
        self.assertEqual(removed, 3)
        self.assertNotIn("Suggested Articles", cleaned)
        self.assertIn("Q: Why does", cleaned)

    def test_removes_promotion_and_tail_author_bio(self) -> None:
        body = """# Topic

Medical paragraph remains.

This article has been reprinted with permission from DogWatch. When you become a member, you receive a free subscription.

Dr. Aly Cohen is an extension veterinarian and clinical instructor at Cornell.
"""
        cleaned, removed = builder.remove_promotional_blocks(body)
        self.assertEqual(removed, 2)
        self.assertEqual(cleaned, "# Topic\n\nMedical paragraph remains.")

    def test_unwraps_outlook_safelink(self) -> None:
        wrapped = (
            "https://nam12.safelinks.protection.outlook.com/"
            "?url=https%3A%2F%2Fwww.vet.cornell.edu%2Ftopic&data=x"
        )
        cleaned = builder.clean_safelinks(f"[topic]({wrapped})")
        self.assertEqual(cleaned, "[topic](https://www.vet.cornell.edu/topic)")


class DeduplicationTests(unittest.TestCase):
    def test_keeps_filename_without_dash_two(self) -> None:
        first = make_document(filename="same.md")
        duplicate = make_document(filename="same-2.md")
        retained, removed = builder.deduplicate_documents([duplicate, first])
        self.assertEqual([item.source_path.name for item in retained], ["same.md"])
        self.assertEqual([item.source_path.name for item in removed], ["same-2.md"])

    def test_same_id_with_different_body_fails(self) -> None:
        first = make_document(body="# Test\n\nFirst.", original_hash="first", filename="a.md")
        second = make_document(body="# Test\n\nSecond.", original_hash="second", filename="b.md")
        with self.assertRaises(builder.CorpusBuildError):
            builder.deduplicate_documents([first, second])


class ChunkingTests(unittest.TestCase):
    def test_preserves_sentence_boundaries_and_merges_short_section(self) -> None:
        sentences = " ".join(f"Sentence {index} has useful medical context." for index in range(70))
        body = f"# Test\n\n## Short\n\nTiny note.\n\n## Long\n\n{sentences}"
        document = make_document(body=body)
        counter = WordCounter()
        chunks = builder.chunk_documents(
            [document], counter, min_tokens=20, target_tokens=55, max_tokens=75, overlap_tokens=8
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(counter.count(chunk.content), 75)
            self.assertRegex(chunk.content.rstrip(), r"[.!?]$")
            self.assertNotEqual(chunk.content.rstrip()[-1], "-")
        self.assertIn("Tiny note.", chunks[0].content)

    def test_ids_are_deterministic(self) -> None:
        document = make_document()
        chunks = [
            builder.FinalChunk(document, ("Test",), "# Test\n\nFirst."),
            builder.FinalChunk(document, ("Test",), "# Test\n\nSecond."),
        ]
        first = builder.final_records(chunks)
        second = builder.final_records(chunks)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["chunk_id"], "cornell_cat_test_001")
        self.assertEqual(first[1]["chunk_id"], "cornell_cat_test_002")


class LocalCorpusIntegrationTests(unittest.TestCase):
    DOG_DIR = Path(r"C:\Users\om\Downloads\dataset\dog")
    CAT_DIR = Path(r"C:\Users\om\Downloads\dataset\cat")

    @unittest.skipUnless(DOG_DIR.is_dir() and CAT_DIR.is_dir(), "local Cornell source corpus not present")
    def test_expected_source_clean_and_dedup_counts(self) -> None:
        documents = builder.load_documents(self.DOG_DIR, self.CAT_DIR)
        self.assertEqual(len(documents), 287)
        resolved = [
            builder.replace(
                document,
                canonical_url=f"https://www.vet.cornell.edu/{document.document_id}",
            )
            for document in documents
        ]
        cleaned, excluded = builder.clean_documents(resolved)
        deduplicated, removed = builder.deduplicate_documents(cleaned)
        self.assertEqual(excluded, ["cornell_dog_big_red_bark_chat"])
        self.assertEqual(len(removed), 4)
        self.assertEqual(len(deduplicated), 282)
        counts = {
            species: sum(document.species == species for document in deduplicated)
            for species in ("dog", "cat")
        }
        self.assertEqual(counts, {"dog": 159, "cat": 123})


if __name__ == "__main__":
    unittest.main()

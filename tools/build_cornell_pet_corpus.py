#!/usr/bin/env python3
"""Build one retrieval-ready Cornell dog/cat JSONL corpus.

The Markdown files are treated as immutable source material.  Every stage before
``build`` runs in memory and prints an explanation; only ``build`` writes the
final JSONL file.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import parse_qs, unquote, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised by the CLI error path
    tiktoken = None


SOURCE_INSTITUTION = "Cornell University College of Veterinary Medicine"
DOG_COLLECTION_URL = (
    "https://www.vet.cornell.edu/departments-centers-and-institutes/"
    "riney-canine-health-center/canine-health-topics"
)
CAT_COLLECTION_URL = (
    "https://www.vet.cornell.edu/departments-centers-and-institutes/"
    "cornell-feline-health-center/health-information/feline-health-topics"
)
USER_AGENT = "PetCareAI-RAG-CorpusBuilder/1.0 (educational internal project)"
REQUIRED_FIELDS = {
    "id",
    "title",
    "species",
    "source",
    "categories",
    "language",
    "medical_domain",
    "content_hash",
}
EXCLUDED_DOCUMENT_IDS = {"cornell_dog_big_red_bark_chat"}
TITLE_ALIASES = {
    "cornell_cat_ask_elizabeth_are_these_frequent_urinary_tract_infections": "Are These Frequent Urinary Tract Infections?",
    "cornell_cat_ask_elizabeth_care_obese_cats": "Care of Obese Cats",
    "cornell_cat_ask_elizabeth_help_my_cats_killer_what_can_i_do": "Help! My Cat's a Killer; What Can I Do?",
    "cornell_cat_ask_elizabeth_it_time_say_good_bye": "Is It Time to Say Good-bye?",
    "cornell_cat_ask_elizabeth_need_rabies_vaccination_indoor_cats": "Need for Rabies Vaccination for Indoor Cats",
    "cornell_cat_ask_elizabeth_patent_ductus_arteriosus": "Patent Ductus Arteriosus",
    "cornell_cat_ask_elizabeth_should_i_consider_pediatric_spay_or_neuter": "Should I Consider a Pediatric Spay or Neuter?",
    "cornell_cat_ask_elizabeth_what_mrsa": "Resistant Staph Infections",
    "cornell_cat_ask_elizabeth_what_there_treat_idiopathic_megacolon": "What Is There to Treat Idiopathic Megacolon?",
    "cornell_cat_ask_elizabeth_white_cats_and_blindnessdeafness": "White cats and blindness/deafness",
    "cornell_cat_beware_holiday_hazards": "Holiday Hazards",
    "cornell_cat_dyspnea_difficulty_breathing": "Dyspnea",
    "cornell_cat_ear_mites_tiny_critters_can_pose_major_threat": "Ear Mites",
    "cornell_cat_feline_behavior_problems_aggression": "Aggression",
    "cornell_cat_feline_behavior_problems_destructive_behavior": "Destructive Behavior",
    "cornell_cat_feline_behavior_problems_house_soiling": "House Soiling",
    "cornell_cat_feline_cataracts": "Cataracts",
    "cornell_cat_feline_glaucoma": "Glaucoma",
    "cornell_cat_feline_immunodeficiency_virus_fiv": "Feline Immunodeficiency Virus",
    "cornell_cat_heartworm_cats": "Feline Heartworm Infection: Serious",
    "cornell_cat_oral_cavity_tumors": "Oral Tumors",
    "cornell_cat_ringworm_serious_readily_treatable_affliction": "Ringworm",
    "cornell_cat_ticks_and_your_cat": "Ticks",
    "cornell_cat_toxoplasmosis_cats": "Toxoplasmosis",
    "cornell_cat_zoonotic_disease_what_can_i_catch_my_cat": "Zoonotic Disease",
}
BUILTIN_URL_OVERRIDES = {
    # Cornell's collection contains both a legacy node link and a newer page for
    # these titles.  These explicit choices keep distinct source documents apart.
    "cornell_cat_common_cat_hazards": "https://www.vet.cornell.edu/node/4043",
    "cornell_cat_pancreatitis": "https://www.vet.cornell.edu/node/4024",
    "cornell_cat_grieving_loss_your_cat": (
        "https://www.vet.cornell.edu/departments-centers-and-institutes/"
        "cornell-feline-health-center/health-information/feline-health-topics/grieving-loss-your-cat"
    ),
    "cornell_cat_respiratory_infections": (
        "https://www.vet.cornell.edu/departments-centers-and-institutes/"
        "cornell-feline-health-center/health-information/feline-health-topics/respiratory-infections"
    ),
    "cornell_cat_warning_signs_cancer": (
        "https://www.vet.cornell.edu/departments-centers-and-institutes/"
        "cornell-feline-health-center/health-information/feline-health-topics/warning-signs-cancer"
    ),
}


class CorpusBuildError(RuntimeError):
    """Raised when a safety or reproducibility invariant is violated."""


@dataclass(frozen=True)
class SourceDocument:
    source_path: Path
    raw_id: str
    document_id: str
    title: str
    species: str
    source_center: str
    categories: tuple[str, ...]
    last_updated: str | None
    language: str
    medical_domain: str
    original_content_hash: str
    body: str
    canonical_url: str | None = None
    suggested_titles: tuple[str, ...] = ()
    content_hash: str | None = None


@dataclass(frozen=True)
class Atom:
    path: tuple[str, ...]
    text: str
    overlap: bool = False


@dataclass
class ChunkDraft:
    atoms: list[Atom] = field(default_factory=list)


@dataclass(frozen=True)
class FinalChunk:
    document: SourceDocument
    section_path: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class LinkCandidate:
    species: str
    title: str
    url: str


@dataclass(frozen=True)
class FetchedPage:
    requested_url: str
    final_url: str
    html_text: str
    suggested_titles: tuple[str, ...]


def stage_message(number: int, title: str, plain: str) -> None:
    print(f"\n[단계 {number}] {title}")
    print(f"  쉽게 말하면: {plain}")


def parse_frontmatter(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)\Z", raw, re.S)
    if not match:
        raise CorpusBuildError(f"YAML front matter를 찾을 수 없습니다: {path}")
    metadata_text, body = match.groups()
    metadata: dict[str, object] = {}
    for line in metadata_text.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise CorpusBuildError(f"메타데이터 줄을 해석할 수 없습니다: {path}: {line}")
        key = key.strip()
        value = value.strip()
        if value.startswith("["):
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise CorpusBuildError(f"categories 배열이 잘못되었습니다: {path}") from exc
        elif value.startswith('"') and value.endswith('"'):
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                metadata[key] = value[1:-1]
        else:
            metadata[key] = value
    return metadata, normalize_newlines(body).strip()


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def slugify(value: str) -> str:
    value = html.unescape(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(value).lower()).strip()


def normalize_document_id(raw_id: str, species: str, fallback_title: str) -> str:
    if raw_id.startswith("cornell:"):
        parts = raw_id.split(":", 2)
        if len(parts) != 3:
            raise CorpusBuildError(f"정규화할 수 없는 문서 ID입니다: {raw_id}")
        result = f"cornell_{parts[1]}_{slugify(parts[2])}"
    elif raw_id.startswith("cornell_"):
        result = raw_id
    else:
        result = f"cornell_{species}_{slugify(fallback_title)}"
    if not re.fullmatch(r"cornell_(dog|cat)_[a-z0-9_]+", result):
        raise CorpusBuildError(f"정규화할 수 없는 문서 ID입니다: {raw_id}")
    return result


def load_documents(dog_dir: Path, cat_dir: Path) -> list[SourceDocument]:
    stage_message(1, "원본 읽기와 검사", "요리 전에 재료의 수량과 상태를 확인합니다.")
    documents: list[SourceDocument] = []
    for expected_species, directory in (("dog", dog_dir), ("cat", cat_dir)):
        if not directory.is_dir():
            raise CorpusBuildError(f"입력 폴더가 없습니다: {directory}")
        paths = sorted(directory.glob("*.md"), key=lambda item: item.name.lower())
        for path in paths:
            metadata, body = parse_frontmatter(path)
            missing = sorted(REQUIRED_FIELDS - metadata.keys())
            if missing:
                raise CorpusBuildError(f"필수 메타데이터 누락({', '.join(missing)}): {path}")
            species = str(metadata["species"]).strip()
            if species != expected_species:
                raise CorpusBuildError(
                    f"폴더와 species가 다릅니다: {path} (예상={expected_species}, 실제={species})"
                )
            categories = metadata["categories"]
            if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
                raise CorpusBuildError(f"categories는 문자열 배열이어야 합니다: {path}")
            if not body:
                raise CorpusBuildError(f"본문이 비어 있습니다: {path}")
            raw_id = str(metadata["id"]).strip()
            title = str(metadata["title"]).strip()
            document_id = normalize_document_id(raw_id, species, title)
            last_updated = str(metadata.get("last_updated", "")).strip() or None
            documents.append(
                SourceDocument(
                    source_path=path,
                    raw_id=raw_id,
                    document_id=document_id,
                    title=title,
                    species=species,
                    source_center=str(metadata["source"]).strip(),
                    categories=tuple(str(item).strip() for item in categories),
                    last_updated=last_updated,
                    language=str(metadata["language"]).strip(),
                    medical_domain=str(metadata["medical_domain"]).strip(),
                    original_content_hash=str(metadata["content_hash"]).strip(),
                    body=body,
                )
            )
    by_species = {name: sum(doc.species == name for doc in documents) for name in ("dog", "cat")}
    print(f"  읽은 문서: {len(documents)}개 (개 {by_species['dog']}개, 고양이 {by_species['cat']}개)")
    return documents


def explain_metadata(documents: Sequence[SourceDocument]) -> None:
    stage_message(2, "메타데이터 통일", "서로 다른 이름표를 같은 양식으로 교체합니다.")
    null_dates = sum(doc.last_updated is None for doc in documents)
    print(f"  document_id, species 배열, categories 배열 형식을 {len(documents)}개 문서에 적용했습니다.")
    print(f"  빈 last_updated {null_dates}개는 null로 취급합니다.")
    for doc in documents[:2]:
        print(f"  예시: {doc.raw_id} -> {doc.document_id}")


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_href: str | None = None
        self.current_text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self.current_href = dict(attrs).get("href") or ""
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.current_href is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.current_href is not None:
            title = normalize_space("".join(self.current_text))
            if title and self.current_href:
                self.links.append((title, self.current_href))
            self.current_href = None
            self.current_text = []


class SuggestedArticlesParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.heading_tag: str | None = None
        self.heading_text: list[str] = []
        self.active = False
        self.anchor_depth = 0
        self.anchor_text: list[str] = []
        self.titles: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if re.fullmatch(r"h[1-6]", tag):
            if self.active:
                self.active = False
            self.heading_tag = tag
            self.heading_text = []
        elif tag == "a" and self.active:
            self.anchor_depth += 1
            if self.anchor_depth == 1:
                self.anchor_text = []

    def handle_data(self, data: str) -> None:
        if self.heading_tag:
            self.heading_text.append(data)
        if self.active and self.anchor_depth:
            self.anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.heading_tag == tag:
            heading = normalized_title("".join(self.heading_text))
            self.active = heading == "suggested articles"
            self.heading_tag = None
            self.heading_text = []
        elif tag == "a" and self.active and self.anchor_depth:
            self.anchor_depth -= 1
            if self.anchor_depth == 0:
                title = normalize_space("".join(self.anchor_text))
                if title and title not in self.titles:
                    self.titles.append(title)
                self.anchor_text = []


def fetch_text(url: str, timeout: int = 45, attempts: int = 3) -> tuple[str, str]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
                final_url = response.geturl()
                declared = response.headers.get_content_charset() or ""
                encoding = "utf-8" if declared.lower() in {"", "iso-8859-1", "latin-1"} else declared
                return final_url, raw.decode(encoding, errors="replace")
        except Exception as exc:  # noqa: BLE001 - retry transient URL/network failures
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def canonicalize_cornell_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {"www.vet.cornell.edu", "vet.cornell.edu"}:
        raise CorpusBuildError(f"Cornell 외부 URL은 사용할 수 없습니다: {url}")
    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
    return urlunparse(("https", "www.vet.cornell.edu", path, "", "", ""))


def collect_listing_links() -> list[LinkCandidate]:
    candidates: dict[tuple[str, str, str], LinkCandidate] = {}
    empty_pages = 0
    for page in range(25):
        url = f"{DOG_COLLECTION_URL}?page={page}"
        _, text = fetch_text(url)
        parser = AnchorParser()
        parser.feed(text)
        before = len(candidates)
        for title, href in parser.links:
            absolute = urljoin(url, href)
            parsed = urlparse(absolute)
            if parsed.netloc.lower() not in {"www.vet.cornell.edu", "vet.cornell.edu"}:
                continue
            canonical = canonicalize_cornell_url(absolute)
            candidates[("dog", normalized_title(title), canonical)] = LinkCandidate("dog", title, canonical)
        if len(candidates) == before:
            empty_pages += 1
            if empty_pages >= 2:
                break
        else:
            empty_pages = 0
    _, text = fetch_text(CAT_COLLECTION_URL)
    parser = AnchorParser()
    parser.feed(text)
    for title, href in parser.links:
        absolute = urljoin(CAT_COLLECTION_URL, href)
        parsed = urlparse(absolute)
        if parsed.netloc.lower() not in {"www.vet.cornell.edu", "vet.cornell.edu"}:
            continue
        canonical = canonicalize_cornell_url(absolute)
        candidates[("cat", normalized_title(title), canonical)] = LinkCandidate("cat", title, canonical)
    return sorted(candidates.values(), key=lambda item: (item.species, item.url, item.title))


def url_slug(url: str) -> str:
    return slugify(unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]))


def document_slug(document: SourceDocument) -> str:
    prefix = f"cornell_{document.species}_"
    return document.document_id.removeprefix(prefix)


def load_url_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"URL override 파일을 읽을 수 없습니다: {path}") from exc
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise CorpusBuildError("URL override는 document_id → URL JSON 객체여야 합니다.")
    return {key: canonicalize_cornell_url(value) for key, value in data.items()}


def match_document_urls(
    documents: Sequence[SourceDocument], candidates: Sequence[LinkCandidate], overrides: dict[str, str]
) -> dict[str, str]:
    slug_index: dict[tuple[str, str], set[str]] = {}
    title_index: dict[tuple[str, str], set[str]] = {}
    for candidate in candidates:
        slug_index.setdefault((candidate.species, url_slug(candidate.url)), set()).add(candidate.url)
        title_index.setdefault((candidate.species, normalized_title(candidate.title)), set()).add(candidate.url)
    result: dict[str, str] = {}
    failures: list[str] = []
    for document in documents:
        if document.document_id in overrides:
            result[document.document_id] = overrides[document.document_id]
            continue
        slug_matches = slug_index.get((document.species, document_slug(document)), set())
        matches = slug_matches
        if not matches:
            lookup_title = TITLE_ALIASES.get(document.document_id, document.title)
            matches = title_index.get((document.species, normalized_title(lookup_title)), set())
        if len(matches) != 1:
            failures.append(
                f"{document.document_id}: 후보 {len(matches)}개"
                + (f" ({', '.join(sorted(matches))})" if matches else "")
            )
        else:
            result[document.document_id] = next(iter(matches))
    if failures:
        preview = "\n  ".join(failures[:30])
        raise CorpusBuildError(
            "URL을 일대일로 확정하지 못했습니다. 검증된 URL override를 추가하세요.\n  " + preview
        )
    return result


def fetch_article_page(url: str) -> FetchedPage:
    final_url, text = fetch_text(url)
    canonical = canonicalize_cornell_url(final_url)
    parser = SuggestedArticlesParser()
    parser.feed(text)
    return FetchedPage(url, canonical, text, tuple(parser.titles))


def resolve_urls(
    documents: Sequence[SourceDocument], overrides_path: Path | None = None, workers: int = 4
) -> list[SourceDocument]:
    stage_message(3, "Cornell 원문 URL 복구", "각 문서에 공식 Cornell 주소표를 붙입니다.")
    candidates = collect_listing_links()
    overrides = {key: canonicalize_cornell_url(value) for key, value in BUILTIN_URL_OVERRIDES.items()}
    overrides.update(load_url_overrides(overrides_path))
    mapping = match_document_urls(documents, candidates, overrides)
    unique_urls = sorted(set(mapping.values()))
    fetched: dict[str, FetchedPage] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_article_page, url): url for url in unique_urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                fetched[url] = future.result()
            except Exception as exc:  # noqa: BLE001 - aggregate network failures
                errors.append(f"{url}: {exc}")
    if errors:
        raise CorpusBuildError("접근 검증에 실패한 Cornell URL이 있습니다.\n  " + "\n  ".join(errors[:30]))
    resolved: list[SourceDocument] = []
    for document in documents:
        page = fetched[mapping[document.document_id]]
        resolved.append(
            replace(
                document,
                canonical_url=page.final_url,
                suggested_titles=page.suggested_titles,
            )
        )
    print(f"  {len(resolved)}개 문서의 URL을 일대일로 확정하고 실제 접근을 확인했습니다.")
    return resolved


def split_blocks(markdown: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", normalize_newlines(markdown)) if block.strip()]


def unwrap_safelink(url: str) -> str:
    parsed = urlparse(html.unescape(url))
    if parsed.netloc.lower() != "nam12.safelinks.protection.outlook.com":
        return url
    target = parse_qs(parsed.query).get("url", [""])[0]
    return unquote(target) if target else url


def clean_safelinks(markdown: str) -> str:
    def replace_link(match: re.Match[str]) -> str:
        label, target = match.groups()
        return f"[{label}]({unwrap_safelink(target)})"

    return re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", replace_link, markdown)


def looks_like_suggested_title(block: str, known_titles: set[str]) -> bool:
    compact = normalize_space(re.sub(r"^[-*]\s+", "", block))
    normalized = normalized_title(compact)
    if normalized in known_titles:
        return True
    if compact.lower().startswith(("video:", "related:")):
        return True
    if "\n" in block or len(compact) > 120:
        return False
    if re.search(r"[.;]", compact):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", compact)
    if not words:
        return False
    title_like = sum(word[0].isupper() or word.lower() in {"a", "an", "and", "for", "from", "in", "of", "the", "to", "with"} for word in words)
    return title_like / len(words) >= 0.7


PROMO_PATTERNS = (
    re.compile(r"reprinted with permission", re.I),
    re.compile(r"become a member", re.I),
    re.compile(r"free subscription", re.I),
    re.compile(r"\b(?:donate now|make a gift|sign up for (?:our|the) newsletter)\b", re.I),
)
AUTHOR_BIO_PATTERN = re.compile(
    r"^(?:Dr\.\s+|[A-Z][A-Za-z'’.\-]+\s+[A-Z]).*"
    r"(?:\bis an?\b|\bserves as\b|\bgraduated from\b|\breceived (?:a|his|her)\b).*"
    r"(?:veterinarian|professor|instructor|director|degree|D\.?V\.?M|Ph\.?D)",
    re.I | re.S,
)


def remove_suggested_articles(body: str, known_titles: set[str]) -> tuple[str, int]:
    blocks = split_blocks(body)
    output: list[str] = []
    removed = 0
    index = 0
    while index < len(blocks):
        if re.fullmatch(r"#{1,6}\s+Suggested Articles(?:\s+and\s+Resources)?\s*", blocks[index], re.I):
            removed += 1
            index += 1
            while index < len(blocks) and looks_like_suggested_title(blocks[index], known_titles):
                removed += 1
                index += 1
            continue
        output.append(blocks[index])
        index += 1
    return "\n\n".join(output).strip(), removed


def remove_promotional_blocks(body: str) -> tuple[str, int]:
    blocks = split_blocks(body)
    kept: list[str] = []
    removed = 0
    for block in blocks:
        if any(pattern.search(block) for pattern in PROMO_PATTERNS):
            removed += 1
        else:
            kept.append(block)
    while kept and AUTHOR_BIO_PATTERN.search(normalize_space(re.sub(r"^#+\s*", "", kept[-1]))):
        kept.pop()
        removed += 1
    return "\n\n".join(kept).strip(), removed


def hash_content(content: str) -> str:
    normalized = re.sub(r"[ \t]+\n", "\n", normalize_newlines(content)).strip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_documents(documents: Sequence[SourceDocument]) -> tuple[list[SourceDocument], list[str]]:
    stage_message(4, "노이즈 제거", "검색에 방해되는 추천 목록, 광고, 구독 문구와 약력을 걷어냅니다.")
    corpus_titles = {normalized_title(document.title) for document in documents}
    cleaned: list[SourceDocument] = []
    excluded: list[str] = []
    suggested_removed = 0
    promo_removed = 0
    changed_examples: list[tuple[str, int, int]] = []
    for document in documents:
        if document.document_id in EXCLUDED_DOCUMENT_IDS:
            excluded.append(document.document_id)
            continue
        before = document.body
        known_titles = corpus_titles | {normalized_title(title) for title in document.suggested_titles}
        body, suggested_count = remove_suggested_articles(before, known_titles)
        body, promo_count = remove_promotional_blocks(body)
        body = clean_safelinks(body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not re.search(r"[A-Za-z]", re.sub(r"^#+.*$", "", body, flags=re.M)):
            raise CorpusBuildError(f"정제 후 의료 본문이 사라졌습니다: {document.document_id}")
        suggested_removed += suggested_count
        promo_removed += promo_count
        if body != before and len(changed_examples) < 5:
            changed_examples.append((document.document_id, len(before), len(body)))
        cleaned.append(replace(document, body=body, content_hash=hash_content(body)))
    print(f"  Suggested Articles 관련 블록 {suggested_removed}개를 제거했습니다.")
    print(f"  홍보·구독·작성자 약력 블록 {promo_removed}개를 제거했습니다.")
    print(f"  전체가 홍보인 문서 {len(excluded)}개를 제외했습니다: {', '.join(excluded)}")
    for document_id, before_size, after_size in changed_examples:
        print(f"  예시: {document_id} ({before_size}자 -> {after_size}자)")
    return cleaned, excluded


def canonical_duplicate_choice(documents: Sequence[SourceDocument]) -> SourceDocument:
    return sorted(
        documents,
        key=lambda doc: (bool(re.search(r"-2\.md$", doc.source_path.name, re.I)), doc.source_path.name.lower()),
    )[0]


def deduplicate_documents(documents: Sequence[SourceDocument]) -> tuple[list[SourceDocument], list[SourceDocument]]:
    stage_message(5, "완전 중복 제거", "같은 내용의 복사본은 한 장만 남깁니다.")
    by_id: dict[str, list[SourceDocument]] = {}
    for document in documents:
        by_id.setdefault(document.document_id, []).append(document)
    retained: list[SourceDocument] = []
    removed: list[SourceDocument] = []
    for document_id, group in sorted(by_id.items()):
        if len(group) == 1:
            retained.append(group[0])
            continue
        original_hashes = {document.original_content_hash for document in group}
        cleaned_hashes = {document.content_hash for document in group}
        if len(original_hashes) != 1 and len(cleaned_hashes) != 1:
            files = ", ".join(str(document.source_path) for document in group)
            raise CorpusBuildError(f"같은 ID에 서로 다른 본문이 있습니다: {document_id}: {files}")
        keep = canonical_duplicate_choice(group)
        retained.append(keep)
        removed.extend(document for document in group if document is not keep)
    by_cleaned_hash: dict[str, list[SourceDocument]] = {}
    for document in retained:
        if document.content_hash is None:
            raise CorpusBuildError(f"정제 content_hash가 없습니다: {document.document_id}")
        by_cleaned_hash.setdefault(document.content_hash, []).append(document)
    final: list[SourceDocument] = []
    for _, group in sorted(by_cleaned_hash.items()):
        if len(group) == 1:
            final.append(group[0])
        else:
            keep = canonical_duplicate_choice(group)
            final.append(keep)
            removed.extend(document for document in group if document is not keep)
    final.sort(key=lambda document: document.document_id)
    print(f"  완전 중복 {len(removed)}개를 제거해 {len(final)}개 문서를 남겼습니다.")
    for document in removed:
        print(f"  제거: {document.source_path.name} ({document.document_id})")
    return final, removed


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        if tiktoken is None:
            raise CorpusBuildError(
                "tiktoken이 설치되지 않았습니다. 'pip install tiktoken==0.13.0'을 실행하세요."
            )
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))


def parse_markdown_atoms(document: SourceDocument, counter: TokenCounter, max_tokens: int) -> list[Atom]:
    blocks = split_blocks(document.body)
    path: list[str] = [document.title]
    atoms: list[Atom] = []
    for block in blocks:
        heading = re.fullmatch(r"(#{1,6})\s+(.+?)\s*", block, re.S)
        if heading and "\n" not in block:
            level = len(heading.group(1))
            title = normalize_space(heading.group(2))
            if level == 1:
                path = [document.title]
            else:
                path = path[: level - 1] + [title]
            continue
        atoms.extend(split_oversized_block(tuple(path), block, document.title, counter, max_tokens))
    if not atoms:
        raise CorpusBuildError(f"청킹할 본문이 없습니다: {document.document_id}")
    return atoms


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+(?=(?:[-*]\s+)?[A-Z0-9\"'“‘(])")


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    return parts or [text.strip()]


def render_atoms(title: str, atoms: Sequence[Atom]) -> tuple[tuple[str, ...], str]:
    if not atoms:
        return (title,), f"# {title}"
    common = list(atoms[0].path)
    for atom in atoms[1:]:
        common = common[: len(os.path.commonprefix([common, list(atom.path)]))]
    if not common or common[0] != title:
        common = [title]
    lines = [f"# {title}"]
    previous_path: tuple[str, ...] | None = None
    for atom in atoms:
        if atom.path != previous_path and len(atom.path) > 1:
            breadcrumb = " > ".join(atom.path[1:])
            lines.extend(["", f"## {breadcrumb}"])
        lines.extend(["", atom.text.strip()])
        previous_path = atom.path
    return tuple(common), "\n".join(lines).strip()


def split_oversized_block(
    path: tuple[str, ...], block: str, title: str, counter: TokenCounter, max_tokens: int
) -> list[Atom]:
    atom = Atom(path, block.strip())
    _, rendered = render_atoms(title, [atom])
    if counter.count(rendered) <= max_tokens:
        return [atom]
    sentences = split_sentences(block)
    result: list[Atom] = []
    for sentence in sentences:
        candidate = Atom(path, sentence)
        _, rendered_sentence = render_atoms(title, [candidate])
        if counter.count(rendered_sentence) > max_tokens:
            raise CorpusBuildError(
                f"한 문장이 {max_tokens}토큰을 초과하여 안전하게 자를 수 없습니다: {title}: {sentence[:120]}"
            )
        result.append(candidate)
    return result


def draft_token_count(title: str, draft: ChunkDraft, counter: TokenCounter) -> int:
    return counter.count(render_atoms(title, draft.atoms)[1])


def overlap_atoms(atoms: Sequence[Atom], counter: TokenCounter, budget: int) -> list[Atom]:
    if not atoms:
        return []
    selected: list[Atom] = []
    total = 0
    final_path = atoms[-1].path
    for atom in reversed(atoms):
        if atom.path != final_path:
            break
        tokens = counter.count(atom.text)
        if total + tokens > budget:
            break
        selected.append(replace(atom, overlap=True))
        total += tokens
    return list(reversed(selected))


def pack_atoms(
    document: SourceDocument,
    atoms: Sequence[Atom],
    counter: TokenCounter,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    current = ChunkDraft()
    index = 0
    while index < len(atoms):
        atom = atoms[index]
        candidate = ChunkDraft(current.atoms + [atom])
        candidate_tokens = draft_token_count(document.title, candidate, counter)
        if current.atoms and candidate_tokens > max_tokens:
            drafts.append(current)
            current = ChunkDraft(overlap_atoms(current.atoms, counter, overlap_tokens))
            continue
        current.atoms.append(atom)
        index += 1
        if draft_token_count(document.title, current, counter) >= target_tokens:
            drafts.append(current)
            current = ChunkDraft(overlap_atoms(current.atoms, counter, overlap_tokens))
    if current.atoms:
        non_overlap = [atom for atom in current.atoms if not atom.overlap]
        if non_overlap:
            drafts.append(current)
    return drafts


def merge_short_drafts(
    document: SourceDocument,
    drafts: list[ChunkDraft],
    counter: TokenCounter,
    min_tokens: int,
    max_tokens: int,
) -> list[ChunkDraft]:
    if len(drafts) <= 1:
        return drafts
    index = 0
    while index < len(drafts):
        if draft_token_count(document.title, drafts[index], counter) >= min_tokens:
            index += 1
            continue
        merged = False
        for neighbor in (index + 1, index - 1):
            if not (0 <= neighbor < len(drafts)):
                continue
            if neighbor > index:
                atoms = drafts[index].atoms + [atom for atom in drafts[neighbor].atoms if not atom.overlap]
            else:
                atoms = drafts[neighbor].atoms + [atom for atom in drafts[index].atoms if not atom.overlap]
            candidate = ChunkDraft(atoms)
            if draft_token_count(document.title, candidate, counter) <= max_tokens:
                low, high = sorted((index, neighbor))
                drafts[low : high + 1] = [candidate]
                index = max(0, low - 1)
                merged = True
                break
        if not merged:
            # Rebalance complete atoms; never split a sentence or word.
            if index > 0:
                previous = drafts[index - 1]
                movable = [atom for atom in previous.atoms if not atom.overlap]
                while len(movable) > 1 and draft_token_count(document.title, drafts[index], counter) < min_tokens:
                    moved = movable.pop()
                    new_previous = ChunkDraft(movable.copy())
                    new_current = ChunkDraft([moved] + [atom for atom in drafts[index].atoms if not atom.overlap])
                    if draft_token_count(document.title, new_current, counter) > max_tokens:
                        break
                    if draft_token_count(document.title, new_previous, counter) < min_tokens:
                        break
                    drafts[index - 1] = new_previous
                    drafts[index] = new_current
            index += 1
    return drafts


def chunk_documents(
    documents: Sequence[SourceDocument],
    counter: TokenCounter,
    min_tokens: int = 120,
    target_tokens: int = 500,
    max_tokens: int = 600,
    overlap_tokens: int = 50,
) -> list[FinalChunk]:
    stage_message(6, "문장·문단 기반 재청킹", "긴 글을 의미가 끊기지 않는 검색 카드로 나눕니다.")
    final: list[FinalChunk] = []
    short_documents = 0
    for document in documents:
        atoms = parse_markdown_atoms(document, counter, max_tokens)
        drafts = pack_atoms(document, atoms, counter, target_tokens, max_tokens, overlap_tokens)
        drafts = merge_short_drafts(document, drafts, counter, min_tokens, max_tokens)
        document_tokens = counter.count(render_atoms(document.title, atoms)[1])
        for draft in drafts:
            path, content = render_atoms(document.title, draft.atoms)
            tokens = counter.count(content)
            if tokens > max_tokens:
                raise CorpusBuildError(f"청크가 {max_tokens}토큰을 초과했습니다: {document.document_id}: {tokens}")
            if tokens < min_tokens and document_tokens >= min_tokens:
                raise CorpusBuildError(
                    f"병합되지 않은 짧은 청크가 있습니다: {document.document_id}: {tokens}토큰"
                )
            if tokens < min_tokens:
                short_documents += 1
            final.append(FinalChunk(document, path, content))
    print(f"  {len(documents)}개 문서를 {len(final)}개 청크로 만들었습니다.")
    print(f"  문서 전체가 120토큰 미만이라 짧게 보존한 청크: {short_documents}개")
    return final


def final_records(chunks: Sequence[FinalChunk]) -> list[dict[str, object]]:
    grouped: dict[str, list[FinalChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document.document_id, []).append(chunk)
    records: list[dict[str, object]] = []
    for document_id in sorted(grouped):
        for index, chunk in enumerate(grouped[document_id], 1):
            doc = chunk.document
            if not doc.canonical_url or not doc.content_hash:
                raise CorpusBuildError(f"최종 메타데이터가 완전하지 않습니다: {doc.document_id}")
            records.append(
                {
                    "chunk_id": f"{document_id}_{index:03d}",
                    "document_id": document_id,
                    "title": doc.title,
                    "section_path": list(chunk.section_path),
                    "species": [doc.species],
                    "categories": list(doc.categories),
                    "canonical_url": doc.canonical_url,
                    "last_updated": doc.last_updated,
                    "source_institution": SOURCE_INSTITUTION,
                    "source_center": doc.source_center,
                    "language": doc.language,
                    "medical_domain": doc.medical_domain,
                    "content_hash": doc.content_hash,
                    "content": chunk.content,
                }
            )
    return records


def validate_final_records(
    records: Sequence[dict[str, object]], counter: TokenCounter, expected_docs: int = 282
) -> None:
    if not records:
        raise CorpusBuildError("최종 청크가 없습니다.")
    chunk_ids = [str(record["chunk_id"]) for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise CorpusBuildError("중복 chunk_id가 있습니다.")
    documents = {str(record["document_id"]): record for record in records}
    if len(documents) != expected_docs:
        raise CorpusBuildError(f"최종 문서 수가 예상과 다릅니다: {len(documents)} != {expected_docs}")
    species_counts = {
        species: sum(record["species"] == [species] for record in documents.values())
        for species in ("dog", "cat")
    }
    if species_counts != {"dog": 159, "cat": 123}:
        raise CorpusBuildError(f"종별 문서 수가 예상과 다릅니다: {species_counts}")
    banned = ("suggested articles", "become a member", "free subscription")
    for record in records:
        content = str(record["content"])
        if any(term in content.lower() for term in banned):
            raise CorpusBuildError(f"제거 대상 문구가 남았습니다: {record['chunk_id']}")
        if re.fullmatch(r"\s*#{1,6}[^\n]+\s*", content):
            raise CorpusBuildError(f"제목만 있는 청크입니다: {record['chunk_id']}")
        if counter.count(content) > 600:
            raise CorpusBuildError(f"600토큰 초과 청크입니다: {record['chunk_id']}")
        url = str(record["canonical_url"])
        if urlparse(url).netloc != "www.vet.cornell.edu":
            raise CorpusBuildError(f"Cornell URL이 아닙니다: {url}")


def atomic_write_jsonl(output: Path, records: Sequence[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_pipeline(args: argparse.Namespace) -> int:
    documents = load_documents(args.dog_dir, args.cat_dir)
    if args.stage == "audit":
        return 0
    explain_metadata(documents)
    if args.stage == "metadata":
        return 0
    documents = resolve_urls(documents, args.url_overrides, args.workers)
    if args.stage == "urls":
        return 0
    documents, _ = clean_documents(documents)
    if args.stage == "clean":
        return 0
    documents, _ = deduplicate_documents(documents)
    if args.stage == "dedupe":
        return 0
    counter = TokenCounter(args.encoding)
    chunks = chunk_documents(
        documents,
        counter,
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    records = final_records(chunks)
    validate_final_records(records, counter, args.expected_docs)
    if args.stage == "chunk":
        return 0
    stage_message(8, "통합 JSONL 생성", "모든 검색 카드를 한 상자에 담습니다.")
    atomic_write_jsonl(args.output, records)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"  저장 완료: {args.output}")
    print(f"  청크 수: {len(records)}개")
    print(f"  SHA-256: {digest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dog-dir", type=Path, required=True)
    parser.add_argument("--cat-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("audit", "metadata", "urls", "clean", "dedupe", "chunk", "build"),
        default="build",
    )
    parser.add_argument("--url-overrides", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--target-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--expected-docs", type=int, default=282)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not (0 < args.min_tokens < args.target_tokens <= args.max_tokens):
        parser.error("토큰 기준은 0 < min < target <= max 순서여야 합니다.")
    try:
        return run_pipeline(args)
    except CorpusBuildError as exc:
        print(f"\n[중단] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

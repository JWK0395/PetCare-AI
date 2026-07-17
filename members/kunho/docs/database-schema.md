# PetCare-AI DB Schema

PetCare-AI에서 사용하는 주요 데이터베이스 테이블 구조를 정리한 문서입니다.

## 1. `pets` - PET DB

반려동물의 기본 프로필과 건강 관련 기초 정보를 저장합니다.

| 컬럼 | 타입 | 비고 |
| --- | --- | --- |
| `id` | int PK | 반려동물 고유 ID |
| `name` | str | 이름 |
| `species` | str | 강아지 / 고양이 |
| `breed` | str | 견종 / 묘종 |
| `birth_date` | date | 생년월일 |
| `sex` | str | 수컷 / 암컷 |
| `is_neutered` | bool | 중성화 여부 |
| `weight_kg` | float | 몸무게 |
| `size_class` | str | 소형 / 중형 / 대형 |
| `diseases_medications_allergies` | JSON(list) | 질병, 복용약, 알레르기 |
| `created_at` | datetime | 생성일시 |

## 2. `daily_entries` - 일기장 DB

사용자가 작성한 반려동물 일기 원문과 일기에서 추출한 일일 상태 정보를 저장합니다.

| 컬럼 | 타입 | 비고 |
| --- | --- | --- |
| `id` | int PK | 일기 기록 고유 ID |
| `record_date` | date | 기록 날짜 |
| `pet_id` | int FK -> `pets.id` | 어느 반려동물의 기록인지 |
| `raw_text` | Text | 사용자가 작성한 일기 원문 |
| `food` | str | 식사 상태 |
| `water` | str | 음수 상태 |
| `activity` | str | 활동 상태 |
| `symptom` | str | 증상 |
| `stool` | str | 배변 및 설사 상태 |
| `vomit` | str | 구토 상태 |
| `notes` | Text | 기타사항 |

## 3. `diagnoses` - 진단서 DB

반려동물의 진단서 정보와 업로드된 원본 파일 참조를 저장합니다.

| 컬럼 | 타입 | 비고 |
| --- | --- | --- |
| `id` | int PK | 진단서 고유 ID |
| `pet_id` | int FK -> `pets.id` | 어느 반려동물의 진단서인지 |
| `date` | date | 진단서 발급일 또는 진료일 |
| `hospital` | str | 발급 병원 |
| `diagnosis` | str | 진단명 |
| `content` | Text | 진단 내용 및 기타사항 |
| `original_file_ref` | str | 업로드한 원본 파일명 |

## Relationships

| 관계 | 설명 |
| --- | --- |
| `daily_entries.pet_id` -> `pets.id` | 일기 기록은 특정 반려동물에 연결됩니다. |
| `diagnoses.pet_id` -> `pets.id` | 진단서는 특정 반려동물에 연결됩니다. |

## Harness Fixture Compatibility

The local agent harness reads DB-style fixtures from either a zip file or an unpacked directory. Each table file may be a raw JSON array or an object wrapping that array under the expected key.

| Logical table | Canonical path | Root fallback | Accepted wrapper key |
| --- | --- | --- | --- |
| `pets` | `db/pets.json` | `pets.json` | `pets` |
| `daily_entries` | `db/daily_entries.json` | `daily_entries.json` | `daily_entries` |
| `diagnoses` | `db/diagnoses.json` | `diagnoses.json` | `diagnoses` |
| handoff context override | `api/handoff_contexts.json` | `handoff_contexts.json` | `handoff_contexts` |
| fixture RAG chunks | `rag/chunks.json` | `rag_chunks.json` | `chunks` |

`DataBundleBackendProvider` can derive `medical_background` from `pets.diseases_medications_allergies` by splitting entries with `type=disease`, `type=medication|supplement`, and `type=allergy` into the handoff-context shape used by the graph.

The fake backend also applies the same default window as the graph dependencies: `days=3` for recent daily entries and up to 20 diagnosis records sorted newest first.
## RAG Data Boundary

The Cornell RAG vector store is not part of the application PET DB.

RAG local files:

```text
rag_data/chunks/cornell_pet_health_chunks.jsonl
rag_data/evaluation/cornell_retrieval_gold.jsonl
rag_data/chroma/
```

Boundary rules:

- `rag_data/chunks/` contains public Cornell official-source material only.
- `rag_data/chroma/` is generated local Chroma state and is gitignored; the current OpenAI-backed collection contains 732 Cornell chunks across 282 documents.
- Personal pet profiles, daily entries, diagnoses, owner notes, hospital names, and uploaded document text must remain in the application data layer and must not be embedded into the Cornell vector store.
- The Assessment Graph may use personal context for safety/routing/answer composition, but the Cornell provider receives only a retrieval query plus safe filters such as species. The query is normalized with reusable veterinary English terms before embedding, not stored back into the PET DB.

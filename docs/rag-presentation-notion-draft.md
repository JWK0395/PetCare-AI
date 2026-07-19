# PetCare-AI RAG 발표 PPT 제작용 자료

> 목적: 이 문서는 최종 발표문이 아니라, 팀원이 PPT를 만들 때 바로 가져다 쓸 수 있는 슬라이드 소스팩이다.  
> 각 슬라이드는 `핵심 메시지`, `화면에 넣을 문구`, `시각자료 아이디어`, `발표자 노트`로 나누었다.

---

## 발표 전체 방향

### 발표에서 보여줄 핵심 흐름

```text
1. 왜 RAG가 필요한가
2. PetCare-AI에서 RAG가 어떤 흐름으로 동작하는가
3. 왜 처음 검색 성능이 실패했는가
4. 직접 만든 골든 검색 테스트셋으로 어떻게 평가했는가
5. query rewrite를 넣고 검색 성공률이 어떻게 바뀌었는가
6. hybrid rerank는 왜 기본값으로 채택하지 않았는가
7. 최종적으로 어떤 구조를 선택했고, 다음 개선 방향은 무엇인가
```

### 발표자가 계속 유지해야 할 메시지

- 이 프로젝트는 “LLM이 알아서 의료 답변을 생성하는 서비스”가 아니다.
- 먼저 Cornell 공식자료를 chunk 단위로 검색하고, 검색된 근거만 사용해 답변하는 RAG 구조다.
- 이번 개선의 핵심은 한국어 질문을 영어 Cornell corpus에 맞는 retrieval query로 바꾼 것이다.
- 골든 검색 테스트 12개 기준으로 `Rewrite + dense-only`가 기대 문서를 모두 top-5 안에 포함했다.
- hybrid rerank도 구현·실험했지만, 현재 결과에서는 성능이 낮아 후속 튜닝 대상으로 남겼다.

### 발표 시 절대 과장하면 안 되는 부분

- “의료 QA 정확도 100%”라고 말하면 안 된다.
- 정확한 표현은 “직접 설계한 골든 검색 질문 12개에서 기대 Cornell 문서가 top-5 안에 포함된 비율이 100%”다.
- “Hybrid rerank로 성능을 올렸다”고 말하면 안 된다.
- 정확한 표현은 “Hybrid rerank는 실험했지만 현재 설정에서는 dense-only보다 낮아 기본값에서 제외했다”다.

---

## Slide 1. 발표 제목 / 문제 제기

### 슬라이드 제목 후보

**PetCare-AI: Cornell 공식자료 기반 반려동물 건강 RAG**

### 이 슬라이드의 목적

발표 초반에 “우리가 단순 챗봇이 아니라 공식 근거 기반 RAG를 만들었다”는 문제의식을 전달한다.

### 화면에 넣을 문구

- 반려동물 건강 질문은 신뢰 가능한 근거가 중요함
- LLM 단독 답변은 출처와 안전성 검증이 어려움
- PetCare-AI는 Cornell 공식 수의학 자료를 검색한 뒤, 검색된 근거만 사용해 답변

### 시각자료 아이디어

```text
사용자 질문
  ↓
공식 Cornell 자료 검색
  ↓
근거 기반 한국어 답변 + 출처
```

### 발표자 노트

> 저희 프로젝트는 반려동물 건강 질문에 대해 LLM이 바로 답하는 구조가 아니라, Cornell 공식 수의학 자료를 먼저 검색하고 그 근거 안에서만 답변하도록 만든 RAG 구조입니다. 의료나 건강 정보에서는 답변이 그럴듯한 것보다, 어떤 자료를 근거로 했는지가 중요하다고 봤습니다.

---

## Slide 2. 전체 RAG 파이프라인

### 슬라이드 제목 후보

**전체 흐름: Cornell JSONL에서 출처 포함 답변까지**

### 이 슬라이드의 목적

RAG가 어떤 단계로 동작하는지 큰 그림을 보여준다.

### 화면에 넣을 문구

```text
Cornell 공식자료
→ JSONL chunk
→ text-embedding-3-small
→ ChromaDB dense retrieval
→ gpt-5.4-mini 답변 생성
→ Cornell 출처 포함 응답
```

### 핵심 키워드

- `chunk`: 검색 가능한 근거 단위
- `embedding`: 문장/문서를 숫자 벡터로 변환
- `dense retrieval`: 의미적으로 가까운 chunk 검색
- `SOURCE`: LLM 답변에 들어가는 근거

### 시각자료 아이디어

PPT에서는 가로형 파이프라인을 추천한다.

```text
[Cornell 문서] → [Chunk] → [Embedding] → [ChromaDB] → [Top-k SOURCE] → [GPT 답변 + 출처]
```

### 발표자 노트

> Cornell 자료를 바로 LLM에 넣는 것이 아니라, 먼저 검색 가능한 chunk로 나눕니다. 각 chunk를 embedding해서 ChromaDB에 저장하고, 질문이 들어오면 관련 chunk를 top-k로 검색합니다. 이후 GPT는 검색된 SOURCE만 사용해서 한국어 답변을 생성하고, 최종 응답에는 Cornell 출처가 붙습니다.

---

## Slide 3. Chunk 기준 데이터 설계

### 슬라이드 제목 후보

**문서 전체가 아니라 chunk를 검색 단위로 사용**

### 이 슬라이드의 목적

팀원이 “chunk 기준 테스트셋”을 설명할 수 있게 chunk와 document_id 관계를 정리한다.

### 화면에 넣을 문구

- 긴 Cornell 문서를 검색 가능한 작은 단위인 chunk로 분리
- 각 chunk는 `chunk_id`, `document_id`, `title`, `section_path`, `species`, `canonical_url`, `content`를 가짐
- 평가는 chunk 자체보다 “기대 document_id가 top-k 안에 포함되는가”를 기준으로 함

### 예시 구조

```text
document_id: cornell_dog_xylitol_toxicities
  ├─ chunk_id: cornell_dog_xylitol_toxicities_001
  └─ chunk_id: cornell_dog_xylitol_toxicities_002
```

### 시각자료 아이디어

왼쪽에는 긴 문서 아이콘, 오른쪽에는 카드 여러 장을 둔다.

```text
[Xylitol toxicities 문서]
        ↓
[chunk 001] [chunk 002] ...
        ↓
검색 결과에서는 chunk가 나오지만 평가는 document_id로 묶어서 판단
```

### 발표자 노트

> RAG에서는 문서 전체를 한 번에 검색하기보다, 검색 가능한 작은 카드처럼 chunk를 만듭니다. 실제 검색 결과는 chunk 단위로 나오지만, 저희 평가는 기대 Cornell 문서가 검색 결과 안에 들어왔는지를 보기 위해 document_id 기준으로 성공 여부를 판단했습니다.

---

## Slide 4. 처음 검색이 실패한 이유

### 슬라이드 제목 후보

**초기 실패 원인: 한국어 질문과 영어 corpus의 불일치**

### 이 슬라이드의 목적

왜 query rewrite가 필요했는지 문제를 선명하게 보여준다.

### 화면에 넣을 문구

- 사용자 질문: 한국어
- Cornell corpus: 영어
- 한국어 질문을 그대로 embedding하면 영어 문서와 검색 언어가 맞지 않음
- 결과: 골든 질문 12개 모두 기대 문서 top-5 검색 실패

### 실패 결과

| 설정 | Recall@k | MRR | 평균 검색 지연시간 |
| --- | ---: | ---: | ---: |
| Korean dense-only | 0.000 | 0.000 | 482ms |

### 시각자료 아이디어

```text
한국어 질문 벡터  ──X──  영어 Cornell 문서 벡터
       언어/표현 mismatch
```

### 발표자 노트

> 처음에는 한국어 질문을 그대로 임베딩해 영어 Cornell 문서를 검색했습니다. 하지만 실험 결과 골든 질문 12개 모두 기대 문서를 top-5 안에 찾지 못했습니다. 여기서 문제는 dense retrieval 자체라기보다, 질문 언어와 corpus 언어가 맞지 않는다는 점이었습니다.

---

## Slide 5. 직접 설계한 골든 검색 테스트셋

### 슬라이드 제목 후보

**검색 품질은 골든 질문 12개로 평가**

### 이 슬라이드의 목적

“검색 성공률 100%”가 무엇을 의미하는지 명확히 정의한다.

### 화면에 넣을 문구

- 직접 설계한 12개 검색 평가 질문
- dog/cat 주요 질환·위험 상황 포함
- 각 질문마다 기대 Cornell `document_id` 지정
- 기준: 기대 문서가 top-5 안에 들어오면 PASS

### 테스트셋 예시

| case_id | 질문 | 기대 문서 |
| --- | --- | --- |
| `dog_chocolate` | 강아지가 초콜릿을 먹었을 때 어떤 문제가 생길 수 있어? | `cornell_dog_chocolate_toxicity_...` |
| `dog_xylitol` | 강아지가 자일리톨이 들어간 껌을 먹으면 위험한가? | `cornell_dog_xylitol_toxicities` |
| `cat_chronic_kidney` | 고양이 만성 신장질환의 증상과 관리에 대해 알고 싶어 | `cornell_cat_chronic_kidney_disease` |
| `cat_asthma` | 고양이가 기침하고 숨쉬기 힘들어하면 천식일 수 있나? | `cornell_cat_feline_asthma_*` |

### 평가 지표

- `Recall@k`: 기대 문서가 top-k 안에 들어온 질문 비율
- `MRR`: 기대 문서가 몇 위에 나왔는지 반영
- 평균 검색 지연시간: 검색 비용과 속도 확인

### 발표자 노트

> 검색 성능은 답변 문장만 보고 판단하지 않았습니다. 먼저 답변에 필요한 Cornell 문서가 검색 결과에 들어오는지 봤습니다. 각 질문마다 기대 document_id를 정해두고, 그 문서가 top-5 안에 있으면 검색 성공으로 판단했습니다.

---

## Slide 6. Query rewrite 방식

### 슬라이드 제목 후보

**질문 임베딩 전, 영어 수의학 retrieval query로 변환**

### 이 슬라이드의 목적

성능 향상 방식이 “단어를 막 붙인 것”이 아니라 검색용 query rewrite임을 설명한다.

### 화면에 넣을 문구

- 한국어 원문 질문은 사용자 경험과 답변 생성에 유지
- 검색 전 단계에서 영어 retrieval query 생성
- query에는 동물 종, 질병명, 독성 물질, 증상, 응급성, Cornell veterinary health 맥락 포함
- rewrite된 query를 embedding하여 ChromaDB dense 검색

### 예시

| 원문 질문 | retrieval query |
| --- | --- |
| 강아지가 자일리톨이 들어간 껌을 먹으면 위험한가? | `dog xylitol gum poisoning danger emergency symptoms Cornell veterinary health` |
| 고양이 만성 신장질환의 증상과 관리에 대해 알고 싶어 | `cat chronic kidney disease symptoms management Cornell veterinary health` |

### 시각자료 아이디어

```text
한국어 질문
  "강아지가 자일리톨 껌을 먹으면 위험한가?"
        ↓ query rewrite
영어 retrieval query
  "dog xylitol gum poisoning danger emergency symptoms Cornell veterinary health"
        ↓ embedding
ChromaDB 검색
```

### 발표자 노트

> 성능 개선의 핵심은 질문을 검색기가 이해하기 좋은 형태로 바꾸는 것이었습니다. 사용자는 한국어로 질문하지만, Cornell corpus는 영어이기 때문에 검색 전용 query는 영어로 바꿨습니다. 이 query에는 자일리톨, 심장사상충, 췌장염처럼 검색에 중요한 수의학 용어가 포함됩니다.

---

## Slide 7. 실험 결과: 검색 성공률 100%

### 슬라이드 제목 후보

**Query rewrite 후 골든 검색 테스트 12/12 통과**

### 이 슬라이드의 목적

성능 개선 결과를 숫자로 보여준다.

### 화면에 넣을 표

| 설정 | Recall@k | MRR | 평균 검색 지연시간 | 결과 |
| --- | ---: | ---: | ---: | --- |
| Korean dense-only | 0.000 | 0.000 | 482ms | 0/12 |
| Rewrite + dense-only | 1.000 | 0.847 | 1578ms | 12/12 |
| Rewrite + hybrid-rerank | 0.917 | 0.757 | 1642ms | 11/12 |

### 강조 문구

> `Rewrite + dense-only`는 골든 질문 12개 모두에서 기대 Cornell 문서를 top-5 안에 포함했다.

### 시각자료 아이디어

막대그래프 추천:

```text
Recall@k
Korean dense-only          0.000
Rewrite + dense-only       1.000
Rewrite + hybrid-rerank    0.917
```

### 발표자 노트

> 결과를 보면 query rewrite의 효과가 가장 컸습니다. 한국어 질문을 그대로 검색했을 때는 Recall@k가 0이었지만, 영어 retrieval query로 바꾼 뒤 dense 검색을 하자 Recall@k가 1.000이 되었습니다. 이때 검색 성공률 100%는 전체 의료 QA 정확도가 아니라, 직접 만든 골든 검색 질문 12개에서 기대 문서가 top-5 안에 들어왔다는 의미입니다.

---

## Slide 8. Hybrid rerank 실험 결과

### 슬라이드 제목 후보

**Hybrid rerank는 구현했지만 기본값으로 채택하지 않음**

### 이 슬라이드의 목적

BM25/Hybrid를 실험했지만 왜 기본값에서 제외했는지 설명한다.

### 화면에 넣을 문구

- hybrid rerank 방식:
  - dense 후보를 더 넓게 가져옴
  - dense similarity + BM25-style lexical score로 재정렬
- 기대:
  - 정확한 병명·독성 물질·약물명 매칭 보완
- 실제:
  - `Rewrite + dense-only`: 12/12
  - `Rewrite + hybrid-rerank`: 11/12
- 결론:
  - 현재 설정에서는 hybrid가 추가 개선이 아니었음
  - 후속 튜닝 대상으로 유지

### 왜 11/12로 낮아졌나

- rewrite query 안에는 중요한 의학 용어뿐 아니라 일반 단어도 포함됨
- 예: `dog`, `signs`, `veterinary`, `care`, `health`, `articles`
- 이런 일반 단어가 여러 문서에 넓게 등장해 lexical score를 흔들 수 있음
- 실제로 `dog_vomiting` case에서 기대 문서가 top-5 밖으로 밀림

### 발표자 노트

> Hybrid rerank는 정확한 단어 매칭을 보완하기 위해 구현했습니다. 하지만 현재 query에는 일반적인 단어도 많이 포함되어 있어서 BM25-style score가 오히려 일부 문서를 잘못 끌어올렸습니다. 그래서 이번 버전에서는 실험 옵션으로 남기고, 기본값은 query rewrite + dense-only로 정했습니다.

---

## Slide 9. 최종 채택 구조

### 슬라이드 제목 후보

**최종 안정 기본값: Query rewrite + Dense retrieval**

### 화면에 넣을 문구

```text
사용자 한국어 질문
→ gpt-5.4-mini로 영어 retrieval query 생성
→ text-embedding-3-small 질문 임베딩
→ ChromaDB dense retrieval
→ top-k Cornell SOURCE
→ gpt-5.4-mini 한국어 답변 생성
→ Cornell 출처 포함 응답
```

### 최종 선택 이유

- 한국어 UX 유지
- 영어 Cornell corpus와 검색 언어 정렬
- 골든 검색 테스트 12/12 통과
- hybrid보다 안정적인 현재 성능
- 출처 기반 답변 구조 유지

### 발표자 노트

> 최종적으로 안정 기본값은 query rewrite + dense retrieval입니다. 사용자는 한국어로 질문하고, 내부 검색 단계에서는 영어 retrieval query를 사용합니다. 이렇게 하면 사용자 경험은 유지하면서도 영어 Cornell 문서 검색 품질을 높일 수 있었습니다.

---

## Slide 10. 한계와 다음 개선 방향

### 슬라이드 제목 후보

**이번 결과의 의미와 다음 단계**

### 화면에 넣을 문구

- 골든 질문 12개는 검색 단계 smoke test
- 전체 의료 QA 정확도나 실제 사용자 만족도를 의미하지 않음
- 답변 품질과 출처 적합성 평가는 추가 필요
- hybrid rerank는 핵심 medical term 중심으로 재설계 가능
- query rewrite latency와 비용도 추가 최적화 대상

### 다음 개선 아이디어

```text
현재:
영어 retrieval query 하나로 dense 검색과 hybrid lexical score에 모두 사용

개선:
dense용 natural query와 BM25용 핵심 medical terms를 분리
```

### 발표자 노트

> 이번 결과는 검색 단계가 최소 기준을 통과했다는 의미입니다. 다음 단계에서는 답변 품질과 출처 적합성을 더 확인해야 합니다. Hybrid rerank도 버리는 것이 아니라, 전체 query가 아니라 핵심 medical term만 반영하도록 재설계하면 다시 성능 개선 후보가 될 수 있습니다.

---

## 발표용 한 장 요약

### 한 장에 넣을 내용

**문제**

- 한국어 사용자 질문과 영어 Cornell corpus 사이에 검색 언어 mismatch 발생

**해결**

- 질문 임베딩 전에 영어 수의학 retrieval query로 rewrite

**결과**

- Korean dense-only: 0/12
- Rewrite + dense-only: 12/12
- Rewrite + hybrid-rerank: 11/12

**결론**

- 현재 기본값은 `query rewrite + dense retrieval`
- hybrid rerank는 후속 튜닝 대상

---

## 발표에서 사용할 수 있는 짧은 멘트 모음

### 도입

> 이 프로젝트에서는 LLM이 바로 답하는 방식이 아니라, Cornell 공식 수의학 자료를 먼저 검색하고 그 근거를 바탕으로 답변하는 RAG 구조를 만들었습니다.

### RAG 구조 설명

> 문서를 chunk로 나누고, 각 chunk를 embedding해서 ChromaDB에 저장했습니다. 질문이 들어오면 관련 chunk를 검색하고, 검색된 SOURCE만 사용해 답변합니다.

### 실패 원인 설명

> 처음에는 한국어 질문을 그대로 영어 corpus에 검색했기 때문에, 골든 질문 12개 모두 기대 문서를 찾지 못했습니다.

### 개선 방식 설명

> 그래서 질문 임베딩 전에 한국어 질문을 영어 retrieval query로 rewrite했습니다. 이 query에는 질병명, 독성 물질, 증상처럼 검색에 중요한 수의학 용어가 포함됩니다.

### 결과 설명

> Query rewrite를 적용한 dense 검색은 골든 질문 12개 모두에서 기대 Cornell 문서를 top-5 안에 포함했습니다.

### hybrid 설명

> Hybrid rerank도 구현해 비교했지만, 현재 설정에서는 일반 단어가 lexical score에 영향을 줘 일부 case에서 성능이 낮아졌습니다. 그래서 기본값이 아니라 후속 튜닝 대상으로 남겼습니다.

### 결론

> 현재 PetCare-AI RAG의 안정 기본 구조는 한국어 질문을 영어 retrieval query로 바꾼 뒤 dense retrieval을 수행하고, 검색된 Cornell SOURCE만 사용해 답변하는 방식입니다.

---

## PPT 제작 시 추천 시각자료

### 1. 파이프라인 다이어그램

```text
Cornell JSONL → Chunk → Embedding → ChromaDB → Top-k SOURCE → GPT Answer
```

### 2. 문제 원인 다이어그램

```text
한국어 질문 ──X── 영어 corpus
       ↓
검색 언어 mismatch
```

### 3. Query rewrite 전후 비교

```text
Before:
강아지가 자일리톨 껌을 먹으면 위험한가?

After:
dog xylitol gum poisoning danger emergency symptoms Cornell veterinary health
```

### 4. 성능 비교 그래프

```text
Recall@k
Korean dense-only          0.000
Rewrite + dense-only       1.000
Rewrite + hybrid-rerank    0.917
```

### 5. 최종 선택 카드

```text
채택: Query rewrite + dense retrieval
보류: Hybrid rerank
이유: 골든 검색 테스트 기준 dense-only가 더 안정적
```

---

## 근거 파일

- `README.md`
- `docs/rag-evaluation-criteria.md`
- `rag_data/evaluation/cornell_retrieval_gold.jsonl`
- `tools/manage_cornell_rag_db.py`
- `petcare_rag/pipeline.py`
- `tests/test_manage_cornell_rag_db.py`
- `tests/test_cornell_rag_pipeline.py`

# PetCare-AI RAG 발표자료 초안

> 목적: 팀원이 바로 발표자료를 만들 수 있도록, 현재 PetCare-AI RAG 개발 내용을 “문제의식 → 구현 → 실험 → 결론” 흐름으로 정리한다.

---

## 발표 핵심 메시지 3개

1. PetCare-AI의 RAG는 Cornell 공식 수의학 자료를 chunk 단위로 검색하고, 검색된 근거만 사용해 한국어 답변과 출처를 제공하는 구조다.
2. 직접 설계한 골든 검색 테스트셋에서 `Rewrite + dense-only` 설정이 12개 질문 모두 기대 문서를 top-5 안에 포함해 검색 성공률 100%를 기록했다.
3. 성능 향상의 핵심은 한국어 질문을 그대로 임베딩하지 않고, 영어 Cornell corpus에 맞는 짧은 수의학 retrieval query로 rewrite한 것이다.

> 주의: 여기서 100%는 전체 의료 QA 정확도가 아니라, 직접 만든 골든 검색 질문 12개에서 기대 Cornell 문서가 top-5 안에 포함된 비율이다.

---

## 1. 대략적인 RAG 흐름 소개

### 슬라이드 제목 후보

**“PetCare-AI RAG는 Cornell 공식자료를 근거로 답변한다”**

### 슬라이드 핵심 bullet

- 목적: 반려동물 건강 질문에 대해 공식 Cornell 수의학 자료 기반 답변 제공
- corpus: Cornell University College of Veterinary Medicine의 개·고양이 건강자료
- 데이터 단위: 문서 전체가 아니라 검색 가능한 `chunk`
- 검색 방식: `text-embedding-3-small`로 chunk와 질문을 임베딩한 뒤 ChromaDB에서 dense retrieval
- 생성 방식: `gpt-5.4-mini`가 검색된 SOURCE만 사용해 한국어 답변 생성
- 출처: Cornell 제목, URL, chunk ID를 함께 제공

### 발표용 흐름도

```text
Cornell 공식자료
→ JSONL chunk
→ text-embedding-3-small 문서 임베딩
→ ChromaDB dense index
→ 사용자 질문 입력
→ 영어 retrieval query rewrite
→ 질문 임베딩
→ dense retrieval top-k
→ gpt-5.4-mini 답변 생성
→ Cornell 출처 포함 응답
```

### 발표자가 말할 문장

> 저희 RAG는 새로운 의료 지식을 학습한 모델이 아니라, Cornell 공식 수의학 자료를 검색해서 그 근거 안에서만 답변하도록 만든 구조입니다. 원본 자료는 JSONL chunk로 저장하고, 각 chunk를 OpenAI embedding으로 벡터화한 뒤 ChromaDB에 색인합니다. 사용자가 한국어로 질문하면 검색에 적합한 영어 query로 바꾸고, 그 query를 임베딩해서 관련 chunk를 찾습니다. 최종 답변은 검색된 Cornell SOURCE만 사용해 한국어로 생성하고, 출처 URL과 chunk ID를 함께 제공합니다.

### 구현 사실

- 문서 chunk 원본: `rag_data/chunks/cornell_pet_health_chunks.jsonl`
- 기본 chunk 수: 732개
- embedding 모델: `text-embedding-3-small`
- embedding 차원: 1536
- 검색 DB: ChromaDB
- 답변 모델: `gpt-5.4-mini`
- 안정 기본 검색 설정: `query rewrite + dense-only`

### 해석

- 이 프로젝트의 핵심은 “LLM이 그냥 답하는 것”이 아니라 “공식 자료를 먼저 찾고, 찾은 근거만 사용해 답하는 것”이다.
- 따라서 성능 평가는 답변 문장만 보는 것이 아니라, 먼저 올바른 chunk를 검색했는지부터 봐야 한다.

---

## 2. 직접 설계한 골든 검색 테스트셋과 검색 성공률

### 슬라이드 제목 후보

**“검색 품질은 골든 질문 12개로 먼저 검증했다”**

### 슬라이드 핵심 bullet

- 직접 설계한 골든 검색 테스트셋 사용
- 각 질문마다 기대 `document_id`를 지정
- 평가 기준: 기대 문서가 최종 top-k 안에 들어오는지
- top-k 기본값: 5
- 지표:
  - `Recall@k`: 기대 문서가 top-k 안에 들어온 비율
  - `MRR`: 기대 문서가 몇 위에 나왔는지 반영
  - 평균 검색 지연시간

### 골든 테스트셋 예시

| case_id | 사용자 질문 | 기대 문서 |
| --- | --- | --- |
| `dog_chocolate` | 강아지가 초콜릿을 먹었을 때 어떤 문제가 생길 수 있어? | `cornell_dog_chocolate_toxicity_what_should_i_do_if_my_dog_eats_chocolate` |
| `dog_xylitol` | 강아지가 자일리톨이 들어간 껌을 먹으면 위험한가? | `cornell_dog_xylitol_toxicities` |
| `cat_chronic_kidney` | 고양이 만성 신장질환의 증상과 관리에 대해 알고 싶어 | `cornell_cat_chronic_kidney_disease` |
| `cat_asthma` | 고양이가 기침하고 숨쉬기 힘들어하면 천식일 수 있나? | `cornell_cat_feline_asthma_*` |

### 평가 명령어

```powershell
python tools/manage_cornell_rag_db.py evaluate --no-query-rewrite --no-hybrid-rerank
python tools/manage_cornell_rag_db.py evaluate
python tools/manage_cornell_rag_db.py evaluate --hybrid-rerank
```

### 발표용 결과 표

| 설정 | Recall@k | MRR | 평균 검색 지연시간 | 결과 해석 |
| --- | ---: | ---: | ---: | --- |
| Korean dense-only | 0.000 | 0.000 | 482ms | 한국어 질문을 그대로 영어 corpus에 검색하면 기대 문서를 찾지 못함 |
| Rewrite + dense-only | 1.000 | 0.847 | 1578ms | 영어 retrieval query로 바꾸자 12개 모두 top-5 안에 기대 문서 포함 |
| Rewrite + hybrid-rerank | 0.917 | 0.757 | 1642ms | hybrid rerank는 일부 case에서 오히려 기대 문서를 밀어냄 |

### 발표자가 말할 문장

> 검색 성능은 직접 만든 골든 질문 12개로 평가했습니다. 각 질문마다 정답 문장 자체를 보는 것이 아니라, 답변에 필요한 Cornell 문서가 top-5 검색 결과 안에 들어오는지를 봤습니다. 한국어 질문을 그대로 dense 검색했을 때는 0/12로 실패했습니다. 반면 질문을 영어 retrieval query로 바꾼 뒤 dense 검색을 하자 12/12가 모두 통과했고, Recall@k는 1.000이 나왔습니다.

### “검색 성공률 100%”의 정확한 의미

- `Rewrite + dense-only` 설정에서 골든 질문 12개 모두 기대 `document_id`가 top-5 안에 포함되었다.
- 즉, 답변 생성 전 검색 단계가 최소한 필요한 Cornell 근거 문서를 모두 후보로 가져왔다는 뜻이다.
- 단, 이는 전체 의료 QA 품질을 완전히 대표하는 지표가 아니라 현재 RAG 검색 단계의 smoke test다.

### 구현 사실

- 골든 질문 파일: `rag_data/evaluation/cornell_retrieval_gold.jsonl`
- 평가 함수는 각 case의 `expected_document_ids`와 검색 결과의 `document_id`를 비교한다.
- 검색 결과에는 `rank`, `document_id`, `similarity`, `retrieval_query`가 출력된다.

### 해석

- 검색 단계가 실패하면 LLM이 아무리 좋아도 공식 근거 기반 답변을 만들기 어렵다.
- 이번 실험에서는 검색 성능 병목이 dense retrieval 자체라기보다 “한국어 질문과 영어 corpus의 언어 불일치”에 있었다.

---

## 3. 질문 임베딩 전 영어 수의학 retrieval query로 보강

### 슬라이드 제목 후보

**“성능 개선의 핵심은 질문을 검색기가 이해하기 좋은 형태로 바꾸는 것”**

### 슬라이드 핵심 bullet

- 문제: 사용자는 한국어로 질문하지만 Cornell corpus는 영어
- 기존 방식: 한국어 질문을 그대로 embedding → 영어 문서와 의미 연결이 약함
- 개선 방식: 질문 임베딩 전에 한국어 질문을 영어 retrieval query로 rewrite
- rewrite query에는 동물 종, 질병명, 독성 물질, 증상, 응급성, Cornell veterinary health 맥락 포함
- 최종 답변은 여전히 한국어 원문 질문 기준으로 생성

### 실제 query rewrite 예시

| 원문 질문 | 검색용 retrieval query |
| --- | --- |
| 강아지가 초콜릿을 먹었을 때 어떤 문제가 생길 수 있어? | `dog chocolate poisoning symptoms emergency Cornell veterinary health` |
| 강아지가 자일리톨이 들어간 껌을 먹으면 위험한가? | `dog xylitol gum poisoning danger emergency symptoms Cornell veterinary health` |
| 더운 날 강아지가 심하게 헐떡이고 쓰러지는 것은 열사병 증상일까? | `dog heatstroke symptoms heavy panting collapse emergency` |
| 고양이 만성 신장질환의 증상과 관리에 대해 알고 싶어 | `cat chronic kidney disease symptoms management Cornell veterinary health` |

### 발표자가 말할 문장

> 처음에는 한국어 질문을 그대로 임베딩해서 영어 Cornell 문서를 검색했습니다. 그런데 이 방식은 골든 질문 12개에서 모두 실패했습니다. 그래서 질문 임베딩 전에 한국어 질문을 영어 retrieval query로 바꾸는 단계를 추가했습니다. 예를 들어 “강아지가 자일리톨이 들어간 껌을 먹으면 위험한가?”라는 질문은 “dog xylitol gum poisoning danger emergency symptoms Cornell veterinary health”처럼 검색에 필요한 영어 수의학 용어 중심 query로 바뀝니다. 이 query를 임베딩하자 Cornell 문서와 검색 언어가 맞아져 검색 성공률이 0/12에서 12/12로 개선되었습니다.

### 구현 사실

- query rewrite 모델: `gpt-5.4-mini`
- rewrite 목적: 답변 생성이 아니라 검색용 영어 query 생성
- rewrite 결과 사용 위치:
  - `query_embedding_text(retrieval_query)`
  - `text-embedding-3-small`
  - ChromaDB dense search
- 원문 질문 사용 위치:
  - 최종 답변 prompt
  - 사용자에게 반환되는 response question

### 구현 흐름

```text
사용자 질문: 강아지가 자일리톨이 들어간 껌을 먹으면 위험한가?
↓
gpt-5.4-mini query rewrite
↓
retrieval query: dog xylitol gum poisoning danger emergency symptoms Cornell veterinary health
↓
text-embedding-3-small
↓
ChromaDB dense retrieval
↓
Cornell xylitol toxicities chunk 검색
↓
gpt-5.4-mini 한국어 답변 생성
```

### 해석

- 이 방식은 “도메인 용어를 무작정 덧붙이는 것”이 아니라, 질문의 의미를 영어 Cornell 검색에 맞는 짧은 retrieval query로 바꾸는 방식이다.
- 수의학 도메인 용어는 rewrite 결과 안에 포함된다.
- 사용자는 한국어 경험을 유지하고, 검색기는 영어 공식 corpus에 맞는 query를 받는다.

---

## 4. Hybrid rerank는 왜 기본값에서 제외했나

### 슬라이드 제목 후보

**“Hybrid rerank는 구현했지만, 현재 결과 기준으로는 후속 튜닝 대상”**

### 슬라이드 핵심 bullet

- hybrid rerank 방식:
  - dense 검색으로 후보를 더 넓게 가져옴
  - dense similarity와 BM25-style lexical score를 결합해 재정렬
- 기대 효과:
  - 병명, 독성 물질, 약물명처럼 정확한 단어 매칭 보완
- 실제 결과:
  - `Rewrite + dense-only`: 12/12
  - `Rewrite + hybrid-rerank`: 11/12
- 결론:
  - 현재 구현에서는 hybrid가 추가 개선이 아니었음
  - 기본값은 `Rewrite + dense-only`
  - hybrid는 후속 튜닝 대상으로 유지

### 발표자가 말할 문장

> Hybrid rerank도 구현해서 실험했습니다. 원래 목적은 dense 검색이 놓칠 수 있는 정확한 병명이나 독성 물질을 BM25-style lexical score로 보완하는 것이었습니다. 하지만 현재 설정에서는 `dog_vomiting` case에서 기대 문서가 top-5 밖으로 밀렸고, 전체 결과도 11/12로 dense-only보다 낮았습니다. 그래서 이번 버전에서는 hybrid를 기본 채택하지 않고, 후속 튜닝 과제로 남겼습니다.

### 왜 성능 차이가 발생했나

- `dense-only`는 rewrite된 영어 query 전체의 의미를 기준으로 가장 가까운 chunk를 고른다.
- `hybrid-rerank`는 dense 후보를 넓게 가져온 뒤 query 단어와 chunk 단어의 lexical match를 반영한다.
- 현재 query에는 `dog`, `signs`, `veterinary`, `care`, `health`, `articles`처럼 여러 문서에 넓게 등장하는 일반 단어도 포함된다.
- 이 일반 단어들이 BM25-style score에 반영되면서 일부 case에서 진짜 기대 문서보다 다른 증상·응급 문서가 올라왔다.

### 후속 개선 방향

- 전체 query가 아니라 핵심 medical term 중심으로 lexical score 계산
- 예:
  - 전체 query: `dog repeated vomiting causes signs needing veterinary care Cornell veterinary health articles`
  - 핵심 term 후보: `vomiting`, `repeated vomiting`, `blood in vomit`, `lethargy`, `veterinary care`
- query rewrite 결과를 두 가지로 분리하는 방법도 가능:
  - dense용 자연어 retrieval query
  - BM25용 핵심 medical terms

---

## 5. 발표자료로 바로 옮길 수 있는 슬라이드 구성안

### Slide 1. 문제 정의

- 반려동물 건강 질문은 신뢰 가능한 공식 근거가 중요하다.
- 사용자는 한국어로 질문하지만, Cornell 공식 자료는 영어다.
- 따라서 “답변 생성”보다 먼저 “올바른 근거 검색”이 핵심 문제다.

**말할 문장**

> 저희는 반려동물 건강 질문에 대해 LLM이 임의로 답하는 것이 아니라, Cornell 공식 수의학 자료를 근거로 답변하는 RAG 구조를 만들었습니다.

### Slide 2. 전체 RAG 파이프라인

- Cornell JSONL chunk
- OpenAI embedding
- ChromaDB dense retrieval
- gpt-5.4-mini 답변 생성
- Cornell 출처 제공

**말할 문장**

> 문서를 chunk로 나누고, 각 chunk를 embedding해서 ChromaDB에 저장합니다. 질문이 들어오면 관련 chunk를 검색하고, 검색된 SOURCE만 사용해 한국어 답변을 생성합니다.

### Slide 3. 처음 실패한 이유

- 한국어 질문을 그대로 영어 Cornell corpus에 dense 검색
- 골든 질문 12개 모두 기대 문서 top-5 검색 실패
- 결과: Recall@k 0.000

**말할 문장**

> 처음에는 한국어 질문을 그대로 임베딩해 검색했는데, corpus가 영어라서 검색 언어가 맞지 않았고 12개 골든 질문이 모두 실패했습니다.

### Slide 4. Query rewrite 도입

- 한국어 질문을 영어 retrieval query로 변환
- 동물 종, 질병명, 독성 물질, 증상, 응급성 포함
- 변환된 query를 embedding해 검색

**말할 문장**

> 그래서 질문 임베딩 전에 한국어 질문을 영어 retrieval query로 rewrite했습니다. 사용자는 한국어로 질문하지만, 검색기는 영어 Cornell 문서에 맞는 query를 받게 됩니다.

### Slide 5. 골든 검색 테스트셋

- 직접 설계한 12개 질문
- 각 질문마다 기대 Cornell `document_id` 지정
- 기대 문서가 top-5 안에 들어오는지 평가

**말할 문장**

> 평가는 답변 문장이 아니라 검색 단계부터 봤습니다. 각 질문마다 기대 Cornell 문서를 정해두고, 그 문서가 top-5 안에 들어오는지를 확인했습니다.

### Slide 6. 실험 결과

| 설정 | Recall@k | MRR | 평균 지연시간 |
| --- | ---: | ---: | ---: |
| Korean dense-only | 0.000 | 0.000 | 482ms |
| Rewrite + dense-only | 1.000 | 0.847 | 1578ms |
| Rewrite + hybrid-rerank | 0.917 | 0.757 | 1642ms |

**말할 문장**

> Query rewrite를 적용한 dense 검색은 12개 질문 모두 기대 문서를 top-5 안에 포함했습니다. 반면 hybrid rerank는 현재 설정에서는 오히려 한 case를 밀어내 11/12에 그쳤습니다.

### Slide 7. 최종 선택과 이유

- 최종 안정 기본값: `query rewrite + dense-only`
- hybrid rerank: 구현은 되어 있지만 기본값 제외
- 이유: 현재 측정 기준에서 dense-only가 더 안정적

**말할 문장**

> 그래서 현재 기본값은 query rewrite + dense-only로 정했습니다. Hybrid rerank는 이론적으로 정확한 단어 매칭에 도움이 될 수 있지만, 현재 실험 결과에서는 추가 튜닝이 필요했습니다.

### Slide 8. 한계와 다음 단계

- 골든 질문 12개는 smoke test
- 답변 품질 평가는 별도 필요
- hybrid는 핵심 medical term 중심으로 재설계 가능
- query rewrite 품질과 latency도 추가 분석 필요

**말할 문장**

> 이번 평가는 검색 단계의 최소 기준을 확인한 것입니다. 다음 단계에서는 답변 품질과 출처 적합성을 더 평가하고, hybrid rerank는 일반 단어가 아니라 핵심 medical term 중심으로 다시 튜닝할 수 있습니다.

---

## 6. 발표용 짧은 결론 문장

> PetCare-AI RAG의 핵심 개선은 한국어 질문을 영어 Cornell corpus에 맞는 retrieval query로 rewrite한 것이다. 이 방식으로 골든 검색 테스트셋 12개 모두에서 기대 문서를 top-5 안에 포함했고, Recall@k 1.000을 기록했다. Hybrid rerank도 구현해 비교했지만 현재 설정에서는 성능이 낮아 기본값에서 제외하고 후속 튜닝 대상으로 남겼다.

---

## 7. 발표 시 주의할 표현

### 이렇게 말하면 좋다

- “검색 성공률 100%는 골든 질문 12개에서 기대 문서가 top-5 안에 들어왔다는 의미입니다.”
- “현재 성능 개선의 핵심은 query rewrite입니다.”
- “Hybrid rerank는 구현하고 비교했지만, 현재 설정에서는 dense-only보다 낮아 기본값에서 제외했습니다.”
- “이 평가는 전체 의료 답변 품질이 아니라 검색 단계의 smoke test입니다.”

### 이렇게 말하면 위험하다

- “Hybrid rerank로 성능을 올렸습니다.”
  - 현재 측정값 기준으로는 사실이 아니다.
- “의료 QA 성능이 100%입니다.”
  - 검색 테스트 12개에 대한 top-5 성공률 100%이지, 전체 의료 QA 정확도 100%가 아니다.
- “도메인 용어를 단순히 덧붙였습니다.”
  - 실제 구현은 `gpt-5.4-mini`를 사용한 영어 retrieval query rewrite다.

---

## 8. 근거 파일

- `README.md`
- `docs/rag-evaluation-criteria.md`
- `rag_data/evaluation/cornell_retrieval_gold.jsonl`
- `tools/manage_cornell_rag_db.py`
- `petcare_rag/pipeline.py`
- `tests/test_manage_cornell_rag_db.py`
- `tests/test_cornell_rag_pipeline.py`

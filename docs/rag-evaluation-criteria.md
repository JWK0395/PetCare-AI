# PetCare Cornell RAG 성능 평가 기준

이 문서는 현재 PetCare-AI의 Cornell 기반 RAG를 발표·실험할 때 사용할 평가 기준을 정리한다.

현재 시스템의 핵심 흐름은 다음과 같다.

```text
Cornell JSONL
→ OpenAI 질문 임베딩
→ ChromaDB dense 후보 검색
→ hybrid rerank
→ gpt-5.4-mini 답변 생성
→ Cornell 출처가 포함된 응답
```

따라서 “RAG 성능”은 하나의 숫자로 보지 않고, 검색 품질, 출처 품질, 답변 품질, 속도를 나누어 본다.

## 1. 검색 품질

검색 품질은 LLM이 답변하기 전에 올바른 근거 chunk를 가져왔는지 보는 기준이다.

현재 골든 질문 평가는 `rag_data/evaluation/cornell_retrieval_gold.jsonl`의 질문마다 기대 문서가 최종 top-k 안에 들어오는지 확인한다.

주요 지표는 다음과 같다.

- `Recall@k`: 기대 문서가 최종 top-k 안에 들어온 질문의 비율.
- `MRR`: 기대 문서가 몇 위에 나왔는지 반영하는 지표. 1위면 1.0, 2위면 0.5, 3위면 0.333처럼 낮아진다.
- `species filter`: dog 질문에는 dog chunk만, cat 질문에는 cat chunk만 들어오는지 확인한다.
- `metadata completeness`: title, URL, document_id, 본문이 빠지지 않았는지 확인한다.

발표에서 말할 때는 다음처럼 정리하면 된다.

> 먼저 LLM 답변을 보기 전에, 검색 단계가 올바른 Cornell 문서를 top-k 안에 포함하는지 Recall@k와 MRR로 평가했다.

## 2. dense-only와 hybrid-rerank 비교

현재 품질 강화의 핵심은 외부 reranker 모델을 새로 호출하는 것이 아니라, ChromaDB dense 검색 후보를 더 넓게 가져온 뒤 dense similarity와 BM25 스타일 lexical score를 함께 사용해 최종 순서를 다시 정렬하는 것이다.

비교해야 하는 설정은 두 가지다.

```powershell
python tools/manage_cornell_rag_db.py evaluate --no-hybrid-rerank
python tools/manage_cornell_rag_db.py evaluate
```

- `--no-hybrid-rerank`: ChromaDB dense 검색 결과만 평가한다.
- 기본 실행: dense 후보를 더 넓게 가져온 뒤 hybrid rerank한 최종 top-k를 평가한다.

발표에서는 아직 측정 전 결과를 단정하지 않는다.

> dense-only 대비 hybrid rerank가 Recall@k 또는 MRR을 높이는지 실험으로 확인한다.

## 3. 출처 품질

의료 정보 RAG에서는 답변이 그럴듯한 것만으로 충분하지 않다. Cornell 출처와 연결되어야 한다.

출처 품질은 다음을 본다.

- 답변에 SOURCE 번호 또는 URL이 포함되는가?
- 답변의 핵심 주장과 인용된 source가 실제로 연결되는가?
- 응급 상황, 독성, 만성질환처럼 위험도가 높은 질문에서 Cornell 근거가 빠지지 않는가?

이 기준은 자동 평가만으로 끝내기 어렵다. 현재 단계에서는 자동 검색 평가 뒤에 대표 질문을 수동으로 확인한다.

## 4. 답변 품질

답변 품질은 LLM 생성 결과가 사용자의 질문에 맞게 안전하고 근거 기반으로 작성되었는지 보는 기준이다.

확인 항목은 다음과 같다.

- 질문에 직접 답했는가?
- Cornell 근거에 없는 내용을 단정하지 않았는가?
- 위험 상황에서는 수의사 상담 또는 응급 진료 권고를 포함했는가?
- 사용자가 한국어로 질문했을 때 한국어로 자연스럽게 답했는가?
- 답변이 너무 길거나, 반대로 필요한 조건을 생략하지 않았는가?

## 5. 속도 품질

속도는 검색 정확도와 trade-off가 있다. 후보를 많이 가져오고 rerank를 하면 정확도 가능성은 올라가지만 지연시간이 늘 수 있다.

현재 평가 CLI는 질문별 검색 지연시간과 평균 검색 지연시간을 출력한다.

추가로 발표에서 분리해서 보면 좋은 시간은 다음과 같다.

- 질문 임베딩 생성 시간
- ChromaDB dense 검색 시간
- hybrid rerank 시간
- LLM 답변 생성 시간
- 전체 응답 시간

현재 골든 평가는 검색 단계 중심이므로, LLM 답변 생성 시간은 `tools/run_cornell_rag.py --debug`로 대표 질문을 따로 확인한다.

## 6. 발표용 테스트 절차

발표용으로는 다음 순서를 추천한다.

1. DB 상태 확인

```powershell
python tools/manage_cornell_rag_db.py inspect
```

2. dense-only baseline 평가

```powershell
python tools/manage_cornell_rag_db.py evaluate --no-hybrid-rerank
```

3. hybrid-rerank 평가

```powershell
python tools/manage_cornell_rag_db.py evaluate
```

4. 대표 질문으로 최종 답변 확인

```powershell
python tools/run_cornell_rag.py `
  --question "강아지가 초콜릿을 먹으면 왜 위험해?" `
  --species dog `
  --top-k 5 `
  --debug
```

5. 결과 표 작성

| 설정 | Recall@k | MRR | 평균 검색 지연시간 | 해석 |
| --- | --- | --- | --- | --- |
| dense-only | 측정 예정 | 측정 예정 | 측정 예정 | 기본 ChromaDB dense 검색 |
| hybrid-rerank | 측정 예정 | 측정 예정 | 측정 예정 | dense 후보를 넓게 가져온 뒤 lexical score로 재정렬 |

## 7. 해석할 때 주의할 점

- 골든 질문 12개는 smoke test다. 전체 의료 질의 품질을 완전히 대표하지 않는다.
- Cornell corpus는 영어이고 사용자는 한국어로 질문할 수 있으므로, BM25 스타일 lexical score가 항상 유리하다고 단정하면 안 된다.
- hybrid rerank가 Recall@k를 높여도 지연시간이 증가할 수 있다.
- 검색이 좋아져도 LLM 답변이 항상 좋아지는 것은 아니다. 검색 평가와 답변 평가는 분리해서 봐야 한다.
- 실제 측정 전에는 “성능이 개선됐다”가 아니라 “개선을 목표로 한 실험 설정을 추가했다”고 표현한다.

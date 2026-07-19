# PetCare AI — Cornell 공식자료 RAG

이 저장소의 현재 실행 가능한 기능은 Cornell University College of Veterinary Medicine의
개·고양이 건강자료를 검색하고, OpenAI 모델이 검색된 근거만 사용해 한국어 답변과 출처를
반환하는 RAG 파이프라인이다.

새로운 AI 모델을 학습한 것이 아니다. 각 부품의 관계는 다음과 같다.

```text
Cornell JSONL → OpenAI 질문 임베딩 → ChromaDB dense 검색
              → hybrid rerank → gpt-5.4-mini 답변 생성
              → Cornell 출처가 포함된 응답
```

## 파일을 지도처럼 보기

| 경로 | 역할 | Git 공유 |
| --- | --- | --- |
| `rag_data/chunks/cornell_pet_health_chunks.jsonl` | 검색 카드의 휴대 가능한 원본 | 공유 |
| `rag_data/chroma/` | 원본에서 만든 로컬 검색 색인 | 제외, 서버에서 재생성 |
| `petcare_rag/` | 검색·답변·인용 검사 핵심 코드 | 공유 |
| `tools/` | corpus, DB, CLI, API 실행 명령 | 공유 |
| `tests/` | 기존 기능이 깨지지 않았는지 확인 | 공유 |
| `notebooks/` | 단계별 관찰을 위한 학습 자료 | 공유 |
| `.env`, `.venv/` | 비밀 설정과 개인 Python 환경 | 절대 공유하지 않음 |

LangGraph는 모델이나 데이터베이스가 아니라 여러 Agent의 순서와 분기를 관리하는
도구다. 현재 RAG 자체에는 필요하지 않으며 Safety, Context/Trend, Summary가 실제
함수 또는 API로 준비된 뒤 도입한다.

## 1. 처음 받은 팀원이 확인하기

Python 3.10 이상과 PowerShell을 기준으로 한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-rag.txt
python -m unittest discover -s tests -v
```

가상환경을 다른 폴더에서 복사하면 Python 경로가 깨질 수 있다. 그 경우 `.venv`를
삭제하고 위 명령으로 현재 프로젝트 폴더에서 다시 만든다. `.venv` 자체는 팀원에게
전달하지 않는다.

API 없이 실행되는 단위 테스트는 OpenAI 키가 없어도 통과해야 한다. corpus와 DB를
직접 다루는 자세한 순서는 [rag_data/README.md](rag_data/README.md)를 참고한다.

## 2. Cornell RAG를 순서대로 직접 실행해보기

아래 명령은 전체 흐름을 눈으로 확인하기 위한 최소 실행 순서다.

```text
Cornell JSONL → OpenAI 질문 임베딩 → ChromaDB dense 검색
              → hybrid rerank → gpt-5.4-mini 답변 생성
              → Cornell 출처가 포함된 응답
```

실제 비밀값을 `.env.example`이나 Git에 기록하지 않는다. PowerShell 세션에만 설정한다.

```powershell
$env:OPENAI_API_KEY="OpenAI_API_키"
```

### 2-1. Cornell JSONL 검사

원본 검색 카드가 올바른 JSONL인지 확인하고, OpenAI embedding API 연결도 가볍게 점검한다.

```powershell
python tools/manage_cornell_rag_db.py check
```

정상이라면 청크 수, 입력 SHA-256, DB 경로 쓰기 가능 여부, `text-embedding-3-small`
1536차원 연결 확인이 출력된다.

### 2-2. ChromaDB dense 색인 만들기

`text-embedding-3-small`로 문서 chunk를 임베딩하고 ChromaDB 컬렉션에 저장한다. 기존
이전 임베딩 모델 기반 컬렉션과 모델·차원이 다르므로 처음 전환할 때는 rebuild한다.

```powershell
python tools/manage_cornell_rag_db.py index --rebuild
```

생성되는 기본 컬렉션 이름은 `cornell_pet_health_text_embedding_3_small_1536`이다.
`rag_data/chroma/`는 재생 가능한 로컬 색인이므로 Git에 올리지 않는다.

### 2-3. 색인 상태 확인

```powershell
python tools/manage_cornell_rag_db.py inspect
```

확인할 기준:

```text
전체 청크: 732
컬렉션 설정: embedding_model=text-embedding-3-small
1536차원이 아닌 벡터: 0
```

### 2-4. 검색만 따로 보기

답변 생성 없이 dense 검색 결과만 보고 싶을 때 사용한다.

```powershell
python tools/manage_cornell_rag_db.py query `
  --query "강아지가 초콜릿을 먹으면 왜 위험해?" `
  --species dog `
  --top-k 5
```

이 명령은 한국어 질문을 영어 Cornell 문서 검색에 맞는 retrieval query로 바꾼 뒤,
ChromaDB dense 검색 결과를 보여준다. 현재 골든 질문 기준에서는 이 설정이 안정 기본값이다.
실험적으로 hybrid rerank까지 비교하고 싶으면 `--hybrid-rerank`를 붙인다.

```powershell
python tools/manage_cornell_rag_db.py query `
  --query "강아지가 초콜릿을 먹으면 왜 위험해?" `
  --species dog `
  --top-k 5 `
  --hybrid-rerank
```

한국어 원문 질문을 그대로 검색했을 때와 비교하려면 `--no-query-rewrite`를 붙인다.

```powershell
python tools/manage_cornell_rag_db.py query `
  --query "강아지가 초콜릿을 먹으면 왜 위험해?" `
  --species dog `
  --top-k 5 `
  --no-query-rewrite
```

답변 생성과 Cornell 출처까지 포함한 전체 파이프라인은 다음 CLI를 사용한다.

### 2-5. dense 검색 → GPT 답변까지 보기

```powershell
python tools/run_cornell_rag.py `
  --question "강아지가 초콜릿을 먹으면 왜 위험해?" `
  --species dog `
  --top-k 5 `
  --debug
```

`--debug`를 붙이면 다음 정보를 함께 볼 수 있다.

- 질문 임베딩 프롬프트
- 원문 질문과 검색용 영어 retrieval query
- 검색된 chunk 순위
- OpenAI에 전달되는 SOURCE context
- 최종 인용 번호
- Cornell 공식 출처가 붙은 최종 답변

현재 검색 품질 강화는 두 단계다. 먼저 한국어 사용자 질문을 영어 Cornell corpus에 맞는
검색 질의로 rewrite한다. 골든 질문 기준으로는 이 단계만 적용한 dense 검색이 12/12를
통과했다. hybrid rerank는 dense 후보를 더 넓게 가져온 뒤 dense similarity와 BM25 스타일
lexical score를 섞어 최종 top-k를 다시 정렬하는 실험 옵션이며, 현재 측정에서는 11/12로
dense-only보다 낮아 기본값에서 제외했다.

평가 기준과 발표용 테스트 절차는 `docs/rag-evaluation-criteria.md`에 정리되어 있다.
핵심은 검색 품질, 출처 품질, 답변 품질, 속도를 분리해서 보는 것이다.

### 2-6. 골든 질문 평가

```powershell
python tools/manage_cornell_rag_db.py evaluate
```

이 평가는 query rewrite 후 dense 검색한 최종 top-k 안에 골든 질문의 기대 문서가 들어오는지
확인한다. 출력에는 원문 질문, 검색용 retrieval query, `Recall@k`, `MRR`, 평균 검색 지연시간이
포함된다.

발표용 비교는 다음 세 가지를 권장한다.

```powershell
python tools/manage_cornell_rag_db.py evaluate --no-query-rewrite --no-hybrid-rerank
python tools/manage_cornell_rag_db.py evaluate --no-hybrid-rerank
python tools/manage_cornell_rag_db.py evaluate --hybrid-rerank
```

첫 번째는 한국어 원문 질문을 그대로 dense 검색한 baseline이고, 두 번째는 영어 query rewrite만
적용한 dense 검색이며, 세 번째는 query rewrite와 hybrid rerank를 모두 적용한 실험 결과다.
현재 측정 결과는 다음과 같다.

| 설정 | Recall@k | MRR | 평균 검색 지연시간 |
| --- | ---: | ---: | ---: |
| Korean dense-only | 0.000 | 0.000 | 482ms |
| Rewrite + dense-only | 1.000 | 0.847 | 1578ms |
| Rewrite + hybrid-rerank | 0.917 | 0.757 | 1642ms |

안정 기본 평가를 빠르게 실행하려면 다음처럼 실행한다.

```powershell
python tools/manage_cornell_rag_db.py evaluate
```

답변 문장 품질을 완전히 보장하는 테스트는 아니며, 검색 설정이 최소 기준을 통과하는지
보는 smoke test다.

## 3. RAG 담당자가 공용 서버 준비하기

서버에서도 실제 비밀값을 `.env.example`이나 Git에 기록하지 않는다. PowerShell 세션 또는
배포 환경의 Secret 설정에만 저장한다.

```powershell
$env:OPENAI_API_KEY="OpenAI_API_키"
$env:PETCARE_RAG_SERVICE_TOKEN="팀_서비스용_긴_임의_문자열"
```

처음 한 번 ChromaDB를 만들고 검사한다.

```powershell
python tools/manage_cornell_rag_db.py check
python tools/manage_cornell_rag_db.py index
python tools/manage_cornell_rag_db.py inspect
python tools/manage_cornell_rag_db.py evaluate
```

정상 기준은 732개 청크, 1536차원, 골든 질문 12개 top-5 통과다. ChromaDB는
재생 가능한 색인이므로 Git에 올리지 않는다.

## 4. 공용 HTTP API 실행하기

서버 PC 안에서만 시험할 때:

```powershell
python tools/run_cornell_rag_api.py
```

팀 네트워크 또는 배포 환경에서 받을 때는 네트워크·방화벽 정책을 확인한 뒤 명시적으로
외부 인터페이스에 바인딩한다.

```powershell
python tools/run_cornell_rag_api.py --host 0.0.0.0 --port 8000
```

상태 확인:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
```

질문 호출:

```powershell
$headers = @{ "X-PetCare-Token" = $env:PETCARE_RAG_SERVICE_TOKEN }
$body = @{
  question = "강아지가 초콜릿을 먹으면 왜 위험해?"
  species = "dog"
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/v1/rag/answer `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

팀원은 OpenAI API 키가 아니라 `X-PetCare-Token`만 사용한다. 이 토큰도 프론트엔드
소스에 하드코딩하지 않고 팀 백엔드의 Secret으로 관리한다. 브라우저 앱이 RAG 서버를
직접 호출하지 않고 팀 백엔드를 거쳐 호출하는 구조를 사용한다.

API 문서는 서버 실행 중 `http://localhost:8000/docs`에서 확인할 수 있다.

## 5. PetCare AI에서의 역할 경계

- PET DB, 진단서 DB, 오늘의 상태 DB는 Context/Trend 기능이 읽는다.
- Cornell RAG API는 `question`, `species`, `top_k`만 받는다.
- 개인 기록을 ChromaDB에 넣거나 이 API 요청에 첨부하지 않는다.
- Safety Agent가 응급 여부를 먼저 확인한다.
- Summary Agent가 개인 기록 분석과 Cornell 근거를 마지막에 구분해서 합친다.

세부적인 통합 순서와 향후 LangGraph 상태 계약은
[RAG 통합 가이드](docs/rag-integration-guide.md)에 정리되어 있다.

## 6. Git으로 공유하기

이 기능은 `feature/cornell-rag` 브랜치에서 PR로 공유한다. `.env`, `.venv`,
`rag_data/chroma`가 Git 변경 목록에 보이면 커밋하지 않는다. 원문 corpus는 비공개 팀
저장소의 내부 RAG 용도로만 사용하고 Cornell 출처 URL과 기관 메타데이터를 유지한다.

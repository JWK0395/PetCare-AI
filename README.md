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

## 2. RAG 담당자가 공용 서버 준비하기

실제 비밀값을 `.env.example`이나 Git에 기록하지 않는다. 서버의 PowerShell 세션 또는
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

## 3. 공용 HTTP API 실행하기

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

## 4. PetCare AI에서의 역할 경계

- PET DB, 진단서 DB, 오늘의 상태 DB는 Context/Trend 기능이 읽는다.
- Cornell RAG API는 `question`, `species`, `top_k`만 받는다.
- 개인 기록을 ChromaDB에 넣거나 이 API 요청에 첨부하지 않는다.
- Safety Agent가 응급 여부를 먼저 확인한다.
- Summary Agent가 개인 기록 분석과 Cornell 근거를 마지막에 구분해서 합친다.

세부적인 통합 순서와 향후 LangGraph 상태 계약은
[RAG 통합 가이드](docs/rag-integration-guide.md)에 정리되어 있다.

## 5. Git으로 공유하기

이 기능은 `feature/cornell-rag` 브랜치에서 PR로 공유한다. `.env`, `.venv`,
`rag_data/chroma`가 Git 변경 목록에 보이면 커밋하지 않는다. 원문 corpus는 비공개 팀
저장소의 내부 RAG 용도로만 사용하고 Cornell 출처 URL과 기관 메타데이터를 유지한다.

# Cornell Pet Health RAG corpus

Cornell University College of Veterinary Medicine의 개·고양이 건강 자료를 정제하고 재청킹한 내부 RAG corpus다.

## 데이터 구성

- 최종 입력: `chunks/cornell_pet_health_chunks.jsonl`
- 청크: 732개
- 문서: 282개 (개 159개, 고양이 123개)
- 임베딩 대상: 각 JSON 행의 `content`
- 검색 결과에 보존할 출처: `title`, `canonical_url`, `source_institution`
- 종 필터: `species` 배열의 `dog` 또는 `cat`

이 corpus는 진단·처방·용량 결정이나 응급상황 판단 엔진이 아니다. 검색된 자료는 상담 준비와 공식 자료 인용을 위한 근거로만 사용한다.

## 처음 한 번 준비하기

PowerShell에서 프로젝트 폴더로 이동하고 가상환경을 준비한다.

```powershell
cd C:\Users\om\Documents\NVIDIA_PBL\PetCare-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-rag.txt
```

Google AI Studio에서 API 키를 발급한 다음 현재 PowerShell 창에만 설정한다.

```powershell
$env:GEMINI_API_KEY="발급받은_API_키"
```

API 키를 코드, JSONL, Git commit에 기록하지 않는다. 무료 등급에는 공개 Cornell 문서와 테스트 질문만 전송하고 보호자의 실제 기록이나 개인정보는 전송하지 않는다.

## 순서대로 실행하기

### 1. 재료와 API 연결 검사

```powershell
python tools/manage_cornell_rag_db.py check
```

732개 JSONL, DB 쓰기 권한, Google API 연결, 768차원 임베딩 반환을 검사한다.

### 2. 로컬 ChromaDB 색인 만들기

```powershell
python tools/manage_cornell_rag_db.py index
```

중간에 속도 제한이나 네트워크 문제로 멈추면 같은 명령을 다시 실행한다. 이미 저장된 청크는 건너뛴다. 원본 JSONL이 변경됐다는 오류가 나오면 변경 내용을 먼저 확인한 후에만 새로 만든다.

```powershell
python tools/manage_cornell_rag_db.py index --rebuild
```

`--rebuild`는 기존 컬렉션을 삭제하므로 입력 corpus나 임베딩 모델이 실제로 바뀐 경우에만 사용한다.

### 3. 저장 결과 확인

```powershell
python tools/manage_cornell_rag_db.py inspect
```

이 명령은 Google API를 호출하지 않고 로컬 DB의 청크 수, 종별 수, 벡터 차원과 예시 레코드를 보여준다.

### 4. 한국어 질문으로 검색

```powershell
python tools/manage_cornell_rag_db.py query --species dog --query "강아지가 초콜릿을 먹었어"
python tools/manage_cornell_rag_db.py query --species cat --query "고양이가 소변을 잘 못 봐"
```

검색은 답변을 생성하지 않는다. 관련 Cornell 청크와 제목, 공식 URL, 유사도만 출력한다.

### 5. 골든 질문 12개 평가

```powershell
python tools/manage_cornell_rag_db.py evaluate
```

개 6개와 고양이 6개의 한국어 질문에서 기대 문서가 상위 5개 안에 있는지, 다른 종 문서가 섞이지 않는지 검사한다.

## Cornell 근거 답변 파이프라인

DB 색인과 검색 평가가 끝났다면 검색된 Cornell 청크만 사용해 한국어 답변을 만들 수 있다.

```powershell
python tools/run_cornell_rag.py `
  --species dog `
  --question "강아지가 초콜릿을 먹으면 왜 위험한가?"
```

이 명령은 다음 순서로 작동한다.

1. 한국어 질문을 `gemini-embedding-2`의 768차원 벡터로 만든다.
2. ChromaDB에서 지정한 species의 Cornell 청크만 찾는다.
3. 검색 결과를 `[SOURCE 1]` 형식의 컨텍스트로 조립한다.
4. `gemini-3.5-flash`의 low thinking 설정으로 구조화 답변을 만든다.
5. 모델의 인용 번호를 검사하고 DB에 저장된 Cornell URL만 최종 출처로 표시한다.

중간 과정을 학습하려면 `--debug`를 추가한다. API 키와 전체 벡터는 출력되지 않는다.

```powershell
python tools/run_cornell_rag.py `
  --species cat `
  --question "고양이 만성 신장질환의 흔한 증상은 무엇인가?" `
  --debug
```

프로그램에서 사용하려면 다음 공개 함수를 호출한다.

```python
from petcare_rag import answer_question

response = answer_question(
    question="강아지가 초콜릿을 먹으면 왜 위험한가?",
    species="dog",
    top_k=5,
)
print(response.answer)
for citation in response.citations:
    print(citation.title, citation.url)
```

학습용 노트북은 `notebooks/cornell_rag_walkthrough.ipynb`다. 질문 임베딩, species 필터 전후, top-k, SOURCE 컨텍스트, 근거 부족 처리와 골든 질문 평가를 셀별로 확인할 수 있다.

이 RAG는 일반 건강정보를 공식 자료로 설명하는 모듈이다. 실제 증상의 응급 여부 판정, 개인 기준선 분석, 진단과 처방은 수행하지 않는다. 실제 앱에서는 Safety Agent가 먼저 실행된 뒤 RAG를 호출해야 한다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```

위 명령은 Google API를 호출하지 않는다. 실제 Google 임베딩 통합 테스트까지 실행하려면 API 키를 설정한 상태에서 다음 환경변수를 추가한다.

```powershell
$env:RUN_RAG_INTEGRATION="1"
python -m unittest tests.test_manage_cornell_rag_db.GoogleApiIntegrationTests -v
```

실제 Chroma 검색부터 Gemini 답변과 Cornell 인용까지 통합 검사하려면 다음을 실행한다.

```powershell
$env:RUN_RAG_INTEGRATION="1"
python -m unittest tests.test_cornell_rag_pipeline.GooglePipelineIntegrationTests -v
```

로컬 ChromaDB 파일은 `rag_data/chroma/`에 생성되며 Git에는 포함하지 않는다. 임베딩 모델이나 차원을 변경하면 기존 벡터와 섞지 말고 별도 컬렉션으로 전체 재색인한다.

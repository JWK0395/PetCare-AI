# PetCare AI RAG 통합 가이드

## 현재와 다음 단계

현재 완성된 부품은 공식자료 RAG다. 개인 기록 DB나 응급 판단 시스템까지 완성된 것은
아니다. 구현 순서는 다음과 같이 고정한다.

1. Cornell RAG 코드·JSONL·테스트를 팀 PR로 공유한다.
2. Google 키와 ChromaDB를 가진 공용 RAG API 한 곳을 실행한다.
3. PET DB, 진단서 DB, 오늘의 상태 DB를 읽는 Context/Trend 기능을 별도로 만든다.
4. Safety와 Summary가 실제 함수 또는 API가 된 뒤 LangGraph로 실행 순서를 묶는다.
5. 같은 평가 질문으로 변경 전후 품질을 비교한다.

## 데이터와 책임 분리

```text
PET·진단서·오늘의 상태 DB → Context/Trend ─┐
                                             ├→ Summary → 사용자 응답
사용자 질문 + species       → Cornell RAG ──┘
```

Context/Trend의 출력에는 개인 기록에서 계산한 사실임을 표시한다. RAG의 출력에는
Cornell 제목, URL, chunk ID를 유지한다. Summary는 두 근거를 같은 출처처럼 섞지 않는다.

RAG는 응급 판정, 확정 진단, 약물 용량, 처방 변경을 담당하지 않는다. 개인 기록 원문도
RAG API나 무료 Google API로 전송하지 않는다.

## 향후 LangGraph 계약

LangGraph는 검색 품질을 높이는 도구가 아니라 분기와 상태를 관리하는 오케스트레이터다.
다음 세 가지가 준비되기 전에는 도입하지 않는다.

- Safety, Context/Trend, Summary 중 최소 3개의 호출 가능한 구현
- 각 구현의 입력·출력 JSON 계약
- 응급, 정보 부족, 정상 흐름에 대한 테스트 사례

권장 공통 상태:

```text
pet_id
question
species
safety_status
context_summary
rag_response
final_response
errors
```

권장 분기:

```text
사용자 입력
  → Safety
     ├─ emergency: 일반 RAG·Summary를 건너뛰고 즉시 안내
     └─ non_emergency
          → Context/Trend
             ├─ insufficient_context: 추가 질문
             └─ sufficient_context
                  → Cornell RAG
                  → Summary
```

RAG 노드는 `/v1/rag/answer`를 호출해 `answer`, `citations`,
`insufficient_evidence`만 기록한다. RAG 오류가 발생해도 Cornell 근거를 임의로 생성하지
않고 `errors`에 안전한 오류 코드를 남긴다.

## 통합 완료 기준

- 팀 백엔드는 Google 키 없이 서비스 토큰으로 RAG API를 호출한다.
- 브라우저나 모바일 클라이언트에 Google 키와 서비스 토큰이 노출되지 않는다.
- 응급 흐름에서는 일반 RAG 답변이 사용자 행동 안내보다 먼저 나오지 않는다.
- 개인 기록 기반 문장과 Cornell 기반 문장의 출처가 구분된다.
- 다른 종의 청크가 응답에 포함되지 않는다.
- 근거 부족 시 추측 대신 `insufficient_evidence=true`가 전달된다.

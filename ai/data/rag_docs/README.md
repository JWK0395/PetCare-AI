# RAG 문서 폴더

전문 건강정보 RAG 의 원본 문서를 여기에 둔다.

- 공인 사이트/가이드(WSAVA 등)에서 모은 문서를 텍스트/마크다운/PDF 로 저장
- LLM 으로 분류(식사 관련 / 응급상황 대처 / 질환별 …) 후 임베딩
- `ai/app/rag.py` 의 `RagStore` 가 이 폴더를 인덱싱한다 (TODO)

예시 구조:
```
rag_docs/
  emergency/
    호흡곤란_대처.md
  nutrition/
    식욕부진_가이드.md
  disease/
    슬개골탈구.md
```

# PetCare AI UX/UI Screen Spec

Source: `C:/Users/kyle0/Downloads/petcareai.pdf`  
Version inferred from PDF: `블루화이트 v3 · 7개 화면 · 2026.07.12`

## Document Purpose

This document converts the UX/UI PDF into an AI-friendly Markdown description.
It is intended to help an LLM understand:

- what screens exist
- what each screen is for
- what information is shown
- what actions the user can take
- how the screens connect as a flow
- what product tone and safety logic are implied

## Product Summary

`PetCare AI` is a pet health management app focused on:

- daily health tracking
- AI-assisted symptom assessment
- emergency routing
- medical document intake
- hospital-ready summary generation

The app appears optimized for a single pet profile named `콩이`, but the structure suggests it can generalize to other pets.

## Primary Navigation

Bottom navigation shown repeatedly in the PDF:

- `홈`
- `기록`
- `AI 체크`
- `진료`

Interpretation:

- `홈`: overview dashboard
- `기록`: daily logs and structured health records
- `AI 체크`: symptom triage and status assessment
- `진료`: hospital communication, documents, and care handoff

## Global UX Characteristics

- Tone is calm, clinical, and caregiver-friendly.
- The UI mixes friendly pet context with medical seriousness.
- AI does not act like a final diagnosis tool. It acts like a triage and summarization assistant.
- Safety escalation is explicit when emergency signals appear.
- Structured extraction is used after free-text input or document upload.
- Many screens convert unstructured input into editable structured data.

## Shared Entity Model

### Pet Profile

- Name: `콩이`
- Species context: dog
- Breed: `말티즈 · 순종`
- Birth date: `2021.09.14`
- Age: `만 4세`
- Sex: `수컷`
- Neutered: `완료`
- Weight: `5.08kg`
- Size: `소형견`

### Recurrent Medical Context

- Existing diagnosis: `슬개골 탈구 2기`
- Supplement/medication: `관절 영양제 1일 1회`
- Allergy: `닭고기 알레르기`

## Screen Inventory

The PDF contains 7 screens.

1. Home Dashboard
2. Health Log Diary
3. AI Status Check - General Triage
4. AI Status Check - Emergency Triage
5. Diagnosis Certificate Upload
6. Hospital Handoff Summary
7. Emergency Email Send

---

## 1. Home Dashboard

### Screen Name

`홈 대시보드`

### Purpose

Provide a quick pet overview and current status entry point.

### Key Content

- current date: `7월 11일 금요일`
- section title: `콩이의 오늘`
- pet identity card with breed, birth date, age, sex, neuter status, weight, and size

### UX Role

- serves as the default landing screen
- anchors the app around a single pet
- gives immediate confidence that records are personalized

### Likely Actions

- move to health logging
- move to AI check
- move to treatment/hospital-related screens

### AI-Relevant Notes

- baseline identity and health attributes are highly visible
- these attributes are likely reused in downstream summary and emergency flows

---

## 2. Health Log Diary

### Screen Name

`건강기록 — 일기장`

### Purpose

Allow the owner to describe the pet's day in natural language and let AI extract structured health records.

### Key Content

- date: `7월 11일 금요일`
- title: `오늘의 기록`
- free-text diary input
- AI extraction result summary: `AI 일기에서 4개 기록을 정리했어요`

### Example Free-Text Input

Morning food was left half unfinished. Walk lasted about 20 minutes and the pet seemed less willing to walk than usual. Water intake was normal. One episode of yellow vomit happened in the afternoon.

### Extracted Structured Records

- `식사`: `사료 105g · 평소의 약 50%`
- `음수`: `정상 범위`
- `활동`: `산책 20분 · 평소보다 짧음`
- `증상`: `구토 1회 · 노란색 · 오후`

Each extracted row has a `수정` action, implying editable AI parsing.

### Primary Action

- `기록 저장`

### UX Role

- bridges free-form caregiver memory and structured medical history
- lowers friction for daily tracking
- keeps the human in the loop by allowing edits before save

### AI-Relevant Notes

- important design pattern: unstructured input -> AI extraction -> user correction -> persistence
- this is likely a core data-ingestion pattern across the product

---

## 3. AI Status Check - General Triage

### Screen Name

`AI 상태 체크 — 일반 경로`

### Purpose

Assess non-emergency symptoms using recent history and follow-up questions.

### Key Content

- title: `AI 상태 체크`
- context: `최근 30일 기록 기반`
- user concern input: `오늘 밥을 거의 안 먹고 하루 종일 축 처져 있어요`
- recent trend summary:
  - `식사 ▼32%`
  - `활동 ▼18%`
  - `구토 1회`
- AI conversational response noting the decline started 3 days ago
- follow-up question: `지난 24시간 동안 구토나 설사가 있었나요?`
- user answer: `오후에 노란 토를 한 번 했어요`

### Triage Outcome

`신속 상담 권장`

Supporting rationale:

- `식사량 3일 연속 개인 기준선 30% 이상 미달`
- `기력 저하 + 구토 동반`

### Trust / Safety Copy

- basis mentioned: `WSAVA 보호자 가이드 2024 v2`
- also references `개인 기준선 30일`

### Primary Next Action

- `병원 전달용 요약 만들기`

### UX Role

- functions as a moderate-risk triage flow
- explains reasoning rather than only producing a label
- leads directly into care escalation documentation

### AI-Relevant Notes

- the system uses both static guidance and personalized behavioral baselines
- explanation transparency is a major trust feature

---

## 4. AI Status Check - Emergency Triage

### Screen Name

`AI 상태 체크 — 응급 경로`

### Purpose

Detect emergency symptoms and immediately route the user to urgent care actions.

### Key Content

- title: `AI 상태 체크`
- user concern input: `숨을 가쁘게 몰아쉬고 혀 색이 파래요`
- critical alert: `응급 징후 — 지금 병원에 연락하세요`

### Emergency Hospital List

- `24시 온누리동물의료센터`
- `센트럴동물응급의료센터`

Each hospital entry includes:

- live status such as `진료 중`
- distance such as `1.2km`, `2.8km`
- emergency-related capability notes
- `전화` action

### Additional Actions

- `상태 문서 이메일 전송`

### In-Transit Guidance

- `기도 확보`
- `최대한 안정 유지`
- `음식과 물은 주지 않기`

### UX Role

- emergency-first workflow
- minimal friction and immediate operational actions
- does not continue with long questioning once high-risk signals appear

### AI-Relevant Notes

- escalation threshold is binary and immediate
- this flow is intentionally operational, not exploratory

---

## 5. Diagnosis Certificate Upload

### Screen Name

`진단서 등록`

### Purpose

Upload a veterinary diagnosis PDF and convert it into structured medical data.

### Key Content

- prompt: `동물 진단서 PDF를 올려주세요`
- upload control: `PDF 파일 업로드`
- AI parsing feedback: `AI 진단서에서 5개 항목을 읽었어요`
- uploaded file example: `진단서_행복한동물병원_0702.pdf`

### Extracted Fields

- `문서 종류`: `진단서`
- `발급 병원`: `행복한동물병원 · 김수민 수의사`
- `발급일`: `2026.07.02`
- `진단명`: `슬개골 탈구 2기`
- `처방`: `관절 영양제 1일 1회 · 30일`

Each field again includes `수정`.

### Primary Action

- `진단서 저장`

### UX Role

- another unstructured-to-structured ingestion flow
- extends AI support from symptom logs to formal hospital records

### AI-Relevant Notes

- document parsing feeds longitudinal care context
- this data is reused in hospital handoff and emergency messaging

---

## 6. Hospital Handoff Summary

### Screen Name

`병원 전달용 요약`

### Purpose

Generate a concise, clinician-friendly summary for a hospital visit.

### Key Content

- generated timestamp: `2026.07.11 14:20`
- patient identity summary:
  - `말티즈`
  - `수컷(중성화)`
  - `만 4세`
  - `5.08kg`
- urgency label: `신속 상담`

### Summary Sections

#### 주호소

- `식욕 부진`
- `기력 저하`
- onset: `7/8 시작, 3일째 지속`

#### 경과 · 변화율

- food intake: `30일 기준선 대비 −32% (218→148g/일)`
- weight: `5.30→5.08kg (−4.2%, 2주)`
- activity: `−18%`

#### 동반 증상

- `구토 1회 (노란색, 7/11 오후)`
- `설사 없음`
- `음수 정상`

#### 기존 질환 · 복용약

- `슬개골 탈구 2기`
- `관절 영양제 1일 1회`
- `닭고기 알레르기`

#### 미확인 항목

- `음수량 (7/10 기록 누락)`
- `배뇨 상태`

#### 첨부

- `30일 추세 그래프`
- `혈액검사 6/28`

### Primary Action

- `PDF 저장`

### UX Role

- converts raw pet data into a clinician-ready briefing
- reduces communication burden during stressful visits
- presents both known facts and missing information

### AI-Relevant Notes

- excellent example of retrieval + summarization + risk labeling
- inclusion of unknowns is a strong safety pattern

---

## 7. Emergency Email Send

### Screen Name

`응급 — 상태 문서 이메일 전송`

### Purpose

Send a prefilled emergency email to a hospital with relevant attachments.

### Key Content

- timestamp: `2026.07.11 14:32`
- recipient hospital: `24시 온누리동물의료센터`
- recipient email: `er@onnuri-amc.kr`
- subject:
  - `[응급] 콩이 (말티즈 · 만 4세) — 호흡곤란 · 청색증 의심`

### Auto-Attached Information

- symptom summary: `호흡곤란 · 청색증 · 14:20 발생`
- recent 30-day health records: `식사 · 활동 · 구토`
- medication and allergy: `관절 영양제 · 닭고기`
- diagnosis certificate: `슬개골 탈구 2기 (7/2)`

### Primary Action

- `이메일 전송`

### UX Role

- minimizes caregiver effort during an emergency
- compresses context gathering into a single send action
- supports continuity between AI triage and real-world care

### AI-Relevant Notes

- this is the highest-escalation handoff screen
- the system moves from interpretation into direct operational support

## End-to-End UX Flow

### Normal Monitoring Flow

1. User opens `홈 대시보드`
2. User writes a daily log in `건강기록 — 일기장`
3. AI extracts structured observations
4. User reviews and saves corrected records

### Symptom Escalation Flow

1. User reports a concern in `AI 상태 체크`
2. System checks recent 30-day baseline data
3. System asks follow-up questions
4. System classifies urgency
5. User generates `병원 전달용 요약`
6. User saves or shares the summary

### Emergency Flow

1. User reports severe symptoms
2. System immediately flags emergency risk
3. System shows nearby 24-hour hospitals
4. System offers call and email actions
5. System includes transport guidance
6. User sends a prefilled emergency email with attachments

### Medical Record Intake Flow

1. User uploads a veterinary diagnosis PDF
2. AI extracts structured medical fields
3. User edits as needed
4. User saves the diagnosis record
5. Extracted information becomes available in summary and emergency flows

## Core UX Patterns

### 1. Free Text to Structured Data

Appears in:

- daily diary logging
- diagnosis PDF upload

Pattern:

- user provides messy real-world input
- AI extracts normalized fields
- user confirms or edits
- app stores structured output

### 2. Baseline-Aware Triage

The app does not rely only on absolute symptom severity.
It also compares current behavior with personal historical baselines over 30 days.

Examples:

- food intake down `32%`
- activity down `18%`
- 3-day pattern recognition

### 3. Explainable Risk Output

AI outputs include reasons, not only labels.

Examples:

- why hospital consultation is recommended
- what signals triggered urgency
- what evidence source or guideline is referenced

### 4. Escalation Ladder

Observed care escalation path:

- home monitoring
- structured logging
- AI check
- rapid consultation recommendation
- hospital summary generation
- emergency hospital contact
- emergency email handoff

### 5. Human-in-the-Loop Correction

AI-generated fields are editable before final save.
This reduces OCR/parsing risk and increases trust.

## Information Architecture Summary

### Inputs

- free-text diary notes
- symptom descriptions
- follow-up question answers
- uploaded diagnosis PDFs

### Derived Data

- structured meal/activity/symptom records
- 30-day trend comparisons
- triage outcomes
- hospital-ready summaries
- emergency email payloads

### Outputs

- saved health records
- risk recommendations
- hospital summary PDF
- emergency email

## Safety and Trust Signals

- emergency path is clearly separated from general triage
- consultation advice is evidence-backed and baseline-backed
- unknown or missing data is explicitly surfaced
- direct diagnosis language is avoided in favor of guidance and escalation
- emergency transport instructions are concise and action oriented

## Suggested Structured Interpretation for Future AI Use

If another AI system needs to consume this product spec, it can model the app as:

```yaml
product_name: PetCare AI
core_jobs:
  - track_pet_health_daily
  - triage_symptoms_with_ai
  - ingest_veterinary_documents
  - prepare_hospital_handoff
  - support_emergency_contact
main_entities:
  - pet_profile
  - daily_health_record
  - symptom_report
  - triage_result
  - diagnosis_document
  - hospital_summary
  - emergency_email
critical_patterns:
  - unstructured_to_structured_extraction
  - personalized_baseline_comparison
  - explainable_triage_reasoning
  - editable_ai_output
  - emergency_escalation
```

## One-Line Product Interpretation

This UX/UI concept describes an AI-assisted pet care app that turns daily observations and medical documents into structured health data, triage guidance, and hospital-ready communication artifacts.

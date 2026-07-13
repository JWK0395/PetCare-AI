# PetCare AI 진단서 샘플 생성 구조

이 문서는 PetCare AI에서 진단서 OCR, 문서 요약, 병원 전달용 기록 생성, 샘플 데이터 제작에 사용할 진단서 예시를 구조화하기 위한 문서이다.

참고 이미지의 핵심은 다음 두 가지이다.

- 실제 진단서에는 보호자, 사육 장소, 동물 표시, 병명, 발병/진단/입원/퇴원일, 주요 증상, 치료명칭, 예후 소견, 기타 사항, 병원/수의사 정보가 표 형태로 들어간다.
- 상세 진단서에는 질병명과 진단일뿐 아니라, 내원 경위, 검사 결과, 보호자 설명, 치료 결정, 상태 악화, 사망 또는 예후 같은 서술형 경과가 길게 들어간다.

샘플은 실제 문서처럼 보이게 만드는 것보다, AI가 안정적으로 필드를 추출하고 요약할 수 있도록 만드는 것이 핵심이다.

## 1. 샘플 생성 원칙

진단서 샘플은 아래 요소를 조합해 만든다.

1. 문서 메타데이터
2. 보호자 정보
3. 사육 장소
4. 동물 표시 정보
5. 진단 정보
6. 임상 경과
7. 치료 및 예후
8. 발급 정보
9. 원문형 서술

각 샘플은 다음 기준을 지킨다.

- 실명, 주소, 병원명, 수의사명은 가명 또는 마스킹 값을 사용한다.
- 날짜는 문서 내에서 시간 순서가 맞아야 한다.
- 질병명은 한글명과 영문명을 함께 넣을 수 있다.
- 검사 결과, 치료 경과, 예후는 추정하지 않고 샘플에 명시된 내용만 사용한다.
- 사망, 중증 감염, 면역저하 등 고위험 케이스는 일반 진료 케이스와 분리한다.
- 진단서 문장에는 법적 증명 문구가 포함될 수 있으나, 샘플임을 문서 메타데이터에 명시한다.

## 2. 진단서 필드 구조

### 문서 메타데이터

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| sample_id | 샘플 고유 ID | certificate_skin_scabies_001 |
| document_type | 문서 유형 | diagnosis_certificate |
| source_format | 원본 형태 | scanned_certificate, blank_form, generated_sample |
| language | 문서 언어 | ko |
| sample_notice | 샘플 고지 | 가상 데이터이며 실제 진단서가 아님 |
| extraction_difficulty | 추출 난이도 | low, medium, high |
| layout_type | 레이아웃 유형 | table_form, narrative_form, mixed_form |

### 보호자 및 사육 정보

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| owner_name | 동물 소유자 성명 | 홍길동, 마스킹 |
| owner_address | 동물 소유자 주소 | 서울특별시 ○○구 ○○로 |
| keeper_name | 관리인 성명 | 보호자와 동일 |
| keeping_place | 사육 장소 | 보호자 자택 |

### 동물 표시 정보

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| animal_name | 동물명 | 몽치 |
| species | 종류 | 개, 고양이 |
| breed | 품종 | 말티즈, 믹스견 |
| animal_registration_no | 동물등록번호 | 410000000000000 |
| sex | 성별 | 수컷, 암컷 |
| age | 연령 | 4년 |
| coat_color | 모색 | 흰색 |
| distinctive_features | 특징 | 체중 감소, 피부 병변 |

### 진단 정보

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| disease_name_ko | 병명 한글 | 피부 질환 및 면역저하 |
| disease_name_en | 병명 영문 | Skin Disease & Immunocompromised Disease |
| suspected_disease | 임상적 추정 병명 | 개선충 감염 의심 |
| final_diagnosis | 최종 진단 | 피부 부전각화증, 진균 및 scabies 복합 감염 |
| disease_onset_date | 발병 연월일 | 2021-11-27 |
| diagnosis_date | 진단 연월일 | 2021-11-28 |
| admission_date | 입원일 | 2021-11-27 |
| discharge_date | 퇴원일 | 2021-12-05 |
| death_date | 사망일 | 2021-12-05 |

### 증상, 치료, 예후

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| chief_complaint | 내원 사유 | 심한 소양감과 탈모 |
| major_symptoms | 주요 증상 | 전신 피부 감염, 소양감, 탈모, 쇠약 |
| test_results | 검사 결과 | 면역저하 및 전신 장기 기능 저하 확인 |
| treatment_name | 치료명칭 | 집중 입원 치료, 감염 관리, 대증 치료 |
| treatment_course | 치료 경과 | 보호자 동의 후 입원 치료 결정 |
| prognosis | 예후 소견 | 건강 악화 및 사망 위험 고지 |
| other_notes | 그 밖의 사항 | 보호자에게 상태와 위험성을 설명함 |

### 발급 정보

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| issue_date | 발급일 | 2022-01-17 |
| hospital_name | 동물병원 명칭 | ○○동물병원 |
| hospital_address | 동물병원 주소 | 서울특별시 ○○구 |
| hospital_phone | 전화번호 | 02-000-0000 |
| veterinarian_license_no | 수의사 면허번호 | 제00000호 |
| veterinarian_name | 수의사 성명 | 김수의 |
| signature_type | 서명 방식 | 서명 또는 인 |

## 3. YAML 샘플 기본 템플릿

```yaml
sample_id: short_unique_id
document_type: diagnosis_certificate
source_format: generated_sample
language: ko
sample_notice: 가상 데이터이며 실제 진단서가 아님
extraction_difficulty: low | medium | high
layout_type: table_form | narrative_form | mixed_form

owner:
  owner_name: 마스킹
  owner_address: 마스킹
  keeper_name: 보호자와 동일
  keeping_place: 보호자 자택

animal:
  animal_name: 몽치
  species: 개
  breed: 믹스견
  animal_registration_no: 마스킹
  sex: 수컷
  age: 4년
  coat_color: 미상
  distinctive_features: []

diagnosis:
  disease_name_ko: 피부 질환 및 면역저하
  disease_name_en: Skin Disease & Immunocompromised Disease
  suspected_disease: null
  final_diagnosis: null
  disease_onset_date: null
  diagnosis_date: null
  admission_date: null
  discharge_date: null
  death_date: null

clinical:
  chief_complaint: null
  major_symptoms: []
  test_results: []
  treatment_name: null
  treatment_course: []
  prognosis: null
  other_notes: []

issuer:
  issue_date: null
  hospital_name: 마스킹
  hospital_address: 마스킹
  hospital_phone: 마스킹
  veterinarian_license_no: 마스킹
  veterinarian_name: 마스킹
  signature_type: 서명 또는 인

raw_text: |
  진단서 원문형 서술을 여기에 작성한다.
```

## 4. 표준 서식 기반 샘플

### certificate_blank_form_basic

```yaml
sample_id: certificate_blank_form_basic
document_type: diagnosis_certificate
source_format: blank_form
language: ko
sample_notice: 가상 데이터이며 실제 진단서가 아님
extraction_difficulty: low
layout_type: table_form

owner:
  owner_name: 홍길동
  owner_address: 서울특별시 ○○구 ○○로 00
  keeper_name: 홍길동
  keeping_place: 보호자 자택

animal:
  animal_name: 코코
  species: 개
  breed: 말티즈
  animal_registration_no: 410000000000000
  sex: 수컷
  age: 5년
  coat_color: 흰색
  distinctive_features:
    - 중성화 완료
    - 체중 4.8kg

diagnosis:
  disease_name_ko: 급성 위장염
  disease_name_en: Acute Gastroenteritis
  suspected_disease: 식이성 위장 장애
  final_diagnosis: 급성 위장염
  disease_onset_date: 2026-07-10
  diagnosis_date: 2026-07-11
  admission_date: null
  discharge_date: null
  death_date: null

clinical:
  chief_complaint: 구토 및 식욕 저하
  major_symptoms:
    - 노란색 구토 2회
    - 식욕 저하
    - 복부 불편감
  test_results:
    - 신체검사상 경도 탈수 소견
    - 분변검사상 특이 기생충 소견 없음
  treatment_name: 수액 처치 및 위장관 보호제 투여
  treatment_course:
    - 내원 당일 피하 수액 처치 시행
    - 위장관 보호제와 구토 억제제 처방
    - 24시간 경과 관찰 안내
  prognosis: 약물 치료 후 호전 가능성이 높으나 구토가 반복되면 재내원이 필요함
  other_notes:
    - 보호자에게 식이 제한과 음수 관찰을 안내함

issuer:
  issue_date: 2026-07-11
  hospital_name: PetCare 샘플동물병원
  hospital_address: 서울특별시 ○○구 ○○로 00
  hospital_phone: 02-000-0000
  veterinarian_license_no: 제00000호
  veterinarian_name: 김수의
  signature_type: 서명 또는 인

raw_text: |
  위 동물은 2026년 7월 11일 구토 및 식욕 저하를 주증으로 내원하였으며,
  신체검사상 경도 탈수와 복부 불편감이 확인되었습니다.
  급성 위장염으로 진단하고 수액 처치 및 위장관 보호제 투여를 시행하였으며,
  보호자에게 식이 제한과 증상 반복 시 재내원을 안내하였습니다.
```

## 5. 상세 경과 서술 기반 샘플

### certificate_skin_scabies_critical

```yaml
sample_id: certificate_skin_scabies_critical
document_type: diagnosis_certificate
source_format: scanned_certificate
language: ko
sample_notice: 가상 데이터이며 실제 진단서가 아님
extraction_difficulty: high
layout_type: mixed_form

owner:
  owner_name: 마스킹
  owner_address: 마스킹
  keeper_name: 마스킹
  keeping_place: 보호자 자택

animal:
  animal_name: 몽치
  species: 개
  breed: 미상
  animal_registration_no: 마스킹
  sex: 수컷
  age: 4년
  coat_color: 미상
  distinctive_features:
    - 전신 피부 병변
    - 탈모
    - 심한 소양감

diagnosis:
  disease_name_ko: 피부 질환 및 면역저하
  disease_name_en: Skin Disease & Immunocompromised Disease
  suspected_disease: scabies 감염 의심
  final_diagnosis: 피부 부전각화증, 진균 및 scabies 복합 감염 동반 전신 피부 감염증
  disease_onset_date: 미상
  diagnosis_date: 2021-11-28
  admission_date: 2021-11-27
  discharge_date: null
  death_date: 2021-12-05

clinical:
  chief_complaint: 심한 소양감과 탈모
  major_symptoms:
    - 심한 소양감
    - 탈모
    - 전신 피부 감염
    - 전신 쇠약
  test_results:
    - 면역저하 확인
    - 전신 장기 기능 저하 확인
  treatment_name: 집중 입원 치료 및 감염 관리
  treatment_course:
    - 2021-11-27 보호자 소개로 내원
    - 기존 병원에서 설이와 눈이 보호자 소개로 내원한 경위 확인
    - 보호자 동의 후 집중 치료를 위한 입원 결정
    - 2021-11-28 검사 결과와 건강 악화 위험성을 보호자에게 유선 설명
    - 2021-12-04 보호자에게 상태와 사망 위험성을 고지
    - 면역저하와 전신 쇠약으로 집중 관리 및 치료 시행
  prognosis: 건강 악화로 사망 위험성이 높음을 보호자에게 고지함
  other_notes:
    - 입원이 불가피한 상황으로 판단됨
    - 보호자에게 치료 과정 중 사망 위험성을 설명함
    - 2021-12-05 사망함

issuer:
  issue_date: 2022-01-17
  hospital_name: 마스킹
  hospital_address: 마스킹
  hospital_phone: 마스킹
  veterinarian_license_no: 마스킹
  veterinarian_name: 마스킹
  signature_type: 서명 또는 인

raw_text: |
  2021년 11월 27일 심한 소양감과 탈모로 내원하였으며,
  원장 수의사의 시진상 심한 피부 부전각화증, 진균 및 scabies 복합 감염을 동반한
  전신 피부 감염증과 전신 쇠약으로 진단하였습니다.
  기존 본원에 다니는 환자동물 보호자의 소개로 내원하였고,
  보호자 동의를 받아 집중 치료를 위한 입원 치료를 결정하였습니다.
  2021년 11월 28일 검사 결과 면역저하와 전신 장기 기능 저하가 확인되어
  보호자에게 건강 상태가 심각하며 건강 악화로 위험할 수 있음을 설명하였습니다.
  2021년 12월 4일 보호자 내원 시 상태와 치료 과정 중 사망 위험성을 고지하였고,
  면역저하와 전신 쇠약으로 집중 관리 및 치료를 실시하였으나
  2021년 12월 5일 사망하였습니다.
```

## 6. 경증 외래 진단서 샘플

### certificate_otitis_outpatient

```yaml
sample_id: certificate_otitis_outpatient
document_type: diagnosis_certificate
source_format: generated_sample
language: ko
sample_notice: 가상 데이터이며 실제 진단서가 아님
extraction_difficulty: medium
layout_type: mixed_form

owner:
  owner_name: 이보호
  owner_address: 경기도 ○○시 ○○로 00
  keeper_name: 이보호
  keeping_place: 보호자 자택

animal:
  animal_name: 나비
  species: 고양이
  breed: 코리안 숏헤어
  animal_registration_no: 해당 없음
  sex: 암컷
  age: 3년
  coat_color: 고등어태비
  distinctive_features:
    - 오른쪽 귀 긁음
    - 귀지 증가

diagnosis:
  disease_name_ko: 외이염
  disease_name_en: Otitis Externa
  suspected_disease: 세균성 외이염
  final_diagnosis: 우측 외이염
  disease_onset_date: 2026-07-08
  diagnosis_date: 2026-07-12
  admission_date: null
  discharge_date: null
  death_date: null

clinical:
  chief_complaint: 오른쪽 귀를 자주 긁고 머리를 흔듦
  major_symptoms:
    - 귀 긁음
    - 귀지 증가
    - 외이도 발적
  test_results:
    - 이경 검사상 우측 외이도 발적 확인
    - 귀 분비물 검사상 세균성 염증 의심
  treatment_name: 귀 세정 및 외용약 처방
  treatment_course:
    - 내원 당일 귀 세정 시행
    - 외용 점이제 7일 처방
    - 투약 후 재검 안내
  prognosis: 처방 치료에 반응할 가능성이 높으며 악화 시 재검 필요
  other_notes:
    - 보호자에게 귀 세정 방법과 투약 횟수를 안내함

issuer:
  issue_date: 2026-07-12
  hospital_name: PetCare 샘플동물병원
  hospital_address: 경기도 ○○시 ○○로 00
  hospital_phone: 031-000-0000
  veterinarian_license_no: 제00000호
  veterinarian_name: 박수의
  signature_type: 서명 또는 인

raw_text: |
  위 동물은 2026년 7월 12일 오른쪽 귀를 자주 긁고 머리를 흔드는 증상으로 내원하였습니다.
  이경 검사상 우측 외이도 발적과 분비물 증가가 확인되어 우측 외이염으로 진단하였습니다.
  귀 세정과 외용 점이제 처방을 시행하였으며, 보호자에게 투약 방법과 재검 필요성을 안내하였습니다.
```

## 7. OCR 추출용 체크 포인트

진단서 이미지를 OCR로 처리할 때는 아래 필드를 우선 추출한다.

| 우선순위 | 필드 | 이유 |
| --- | --- | --- |
| 1 | animal_name | 사용자 반려동물 프로필과 연결하기 위함 |
| 1 | disease_name_ko | 진단서 핵심 정보 |
| 1 | diagnosis_date | 진단 시점 확인 |
| 2 | major_symptoms | 병원 방문 사유와 상태 요약에 필요 |
| 2 | treatment_name | 치료 이력 요약에 필요 |
| 2 | prognosis | 보호자 안내와 병원 전달 요약에 필요 |
| 3 | hospital_name | 문서 출처 확인 |
| 3 | veterinarian_name | 발급자 확인 |
| 3 | issue_date | 문서 발급 시점 확인 |

## 8. 병원 전달용 요약 변환 규칙

진단서 샘플을 PetCare AI의 병원 전달용 요약으로 변환할 때는 아래 구조를 사용한다.

```yaml
handoff_summary:
  pet:
    name: animal.animal_name
    species: animal.species
    breed: animal.breed
    sex: animal.sex
    age: animal.age
  diagnosis_history:
    disease: diagnosis.final_diagnosis
    diagnosed_at: diagnosis.diagnosis_date
    hospital: issuer.hospital_name
  clinical_history:
    chief_complaint: clinical.chief_complaint
    symptoms: clinical.major_symptoms
    tests: clinical.test_results
    treatment: clinical.treatment_name
    course: clinical.treatment_course
    prognosis: clinical.prognosis
  notes_for_vet:
    - 원문 진단서에서 확인되는 사실만 요약한다.
    - 보호자가 추가로 입력한 현재 증상과 과거 진단서 내용을 분리한다.
    - 사망, 면역저하, 전신 쇠약, 중증 감염은 고위험 이력으로 태깅한다.
```

## 9. 태그 목록

### 문서 유형

- certificate_table_form
- certificate_narrative_form
- certificate_mixed_form
- scanned_document
- generated_sample

### 질병 및 증상

- skin_disease
- immunocompromised
- scabies
- fungal_infection
- systemic_infection
- gastroenteritis
- vomiting
- diarrhea
- otitis
- itching
- alopecia
- lethargy

### 경과 및 예후

- outpatient
- inpatient
- intensive_care
- recovered
- ongoing_treatment
- poor_prognosis
- death_recorded

### 추출 난이도

- extraction_low
- extraction_medium
- extraction_high
- handwriting_or_scan_noise
- table_boundary_required

## 10. 샘플 작성 템플릿

새 진단서 샘플을 추가할 때는 아래 형식을 사용한다.

```yaml
sample_id: certificate_case_name_001
document_type: diagnosis_certificate
source_format: scanned_certificate | blank_form | generated_sample
language: ko
sample_notice: 가상 데이터이며 실제 진단서가 아님
extraction_difficulty: low | medium | high
layout_type: table_form | narrative_form | mixed_form
tags:
  - certificate_mixed_form
  - outpatient

owner:
  owner_name: 마스킹
  owner_address: 마스킹
  keeper_name: 마스킹
  keeping_place: 보호자 자택

animal:
  animal_name: 샘플명
  species: 개
  breed: 미상
  animal_registration_no: 마스킹
  sex: 미상
  age: 미상
  coat_color: 미상
  distinctive_features: []

diagnosis:
  disease_name_ko: 진단명
  disease_name_en: English Diagnosis Name
  suspected_disease: null
  final_diagnosis: null
  disease_onset_date: null
  diagnosis_date: null
  admission_date: null
  discharge_date: null
  death_date: null

clinical:
  chief_complaint: null
  major_symptoms: []
  test_results: []
  treatment_name: null
  treatment_course: []
  prognosis: null
  other_notes: []

issuer:
  issue_date: null
  hospital_name: 마스킹
  hospital_address: 마스킹
  hospital_phone: 마스킹
  veterinarian_license_no: 마스킹
  veterinarian_name: 마스킹
  signature_type: 서명 또는 인

raw_text: |
  진단서 원문형 문장을 작성한다.
```

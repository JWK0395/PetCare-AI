# PetCare AI 사용자 입력 샘플 구조

이 문서는 PetCare AI에서 사용할 사용자 입력 샘플을 쉽게 만들기 위한 구조화 문서이다.

진단서 예시는 별도 문서에서 다룬다. 여기서는 사용자가 직접 입력하는 정보만 정리한다.

- 반려동물 기본 정보
- 오늘의 상태 기록
- AI 상태 체크 입력
- AI 추가 질문에 대한 사용자 답변
- 병원 전달용 요약 생성을 위해 사용자가 보완하는 정보

## 1. 샘플 생성 원칙

사용자 입력 샘플은 하나의 긴 문장만 만드는 것이 아니라, 아래 요소를 조합해 만든다.

1. 반려동물 프로필
2. 오늘의 상태 기록
3. AI 상태 체크 입력
4. AI 추가 질문 답변
5. 병원 전달용 요약 보완 입력

각 샘플은 다음 관점을 함께 가져야 한다.

- 사용자가 실제로 말할 법한 자연어
- AI가 구조화할 수 있는 명확한 단서
- 정확히 모르는 정보는 추정하지 않는 표현
- 응급 케이스와 일반 관찰 케이스의 분리

## 2. 반려동물 프로필 입력 구조

### 필드

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| name | 반려동물 이름 | 코코 |
| species | 종 | 강아지, 고양이 |
| breed | 견종 또는 묘종 | 말티즈, 푸들, 코리안 숏헤어 |
| birth_date | 생년월일 | 2021-09-14 |
| sex | 성별 | 수컷, 암컷 |
| is_neutered | 중성화 여부 | 완료, 미완료, 모름 |
| weight_kg | 몸무게 | 5.08 |
| diseases | 기존 질병 | 슬개골 탈구 2기 |
| medications | 복용약 또는 영양제 | 관절 영양제 1일 1회 |
| allergies | 알레르기 | 닭고기 알레르기 |
| registered_at | 등록일 | 2026-07-11 |

### 프로필 샘플

```yaml
profile_id: dog_001
name: 코코
species: 강아지
breed: 말티즈
birth_date: 2021-09-14
sex: 수컷
is_neutered: 완료
weight_kg: 5.08
diseases:
  - 슬개골 탈구 2기
medications:
  - 관절 영양제 1일 1회
allergies:
  - 닭고기 알레르기
registered_at: 2026-07-11
```

```yaml
profile_id: cat_001
name: 나비
species: 고양이
breed: 코리안 숏헤어
birth_date: 2020-12-05
sex: 암컷
is_neutered: 완료
weight_kg: 4.2
diseases:
  - 방광염 병력
medications: []
allergies: []
registered_at: 2026-07-11
```

## 3. 오늘의 상태 기록 입력 구조

### 필드

| 필드 | 설명 | 값 예시 |
| --- | --- | --- |
| record_date | 기록 날짜 | 2026-07-11 |
| food | 식사 상태 | 평소처럼 먹음, 절반만 먹음, 거의 안 먹음 |
| water | 음수 상태 | 평소와 비슷함, 평소보다 많음, 거의 안 마심 |
| activity | 활동 상태 | 산책 30분, 산책 거부, 계속 누워 있음 |
| symptoms | 증상 | 구토, 설사, 기력 저하, 절뚝거림 |
| stool | 배변 상태 | 정상 변 1회, 묽은 변 2회, 배변 없음 |
| urination | 배뇨 상태 | 평소와 비슷함, 소변 적음, 색이 진함 |
| weight | 당일 몸무게 | 5.08kg |
| notes | 기타 관찰 | 평소보다 잠을 많이 잠 |
| uncertainty | 불확실한 정보 | 정확한 음수량은 모름 |

### 자연어 템플릿

```text
오늘은 {food}.
물은 {water}.
활동은 {activity}.
배변은 {stool}.
증상은 {symptoms}.
기타로는 {notes}.
{uncertainty}
```

### 값 후보

#### food

- 사료를 평소처럼 거의 다 먹었어요
- 아침 사료를 절반 정도 남겼어요
- 저녁도 평소보다 적게 먹었어요
- 오늘은 간식에도 별로 관심이 없었어요
- 거의 먹지 않았어요

#### water

- 물은 평소처럼 마셨어요
- 물을 평소보다 많이 마신 것 같아요
- 물을 거의 안 마신 것 같아요
- 정확한 양은 모르겠지만 물그릇이 빨리 비었어요

#### activity

- 산책은 아침에 25분 정도 했어요
- 산책은 15분 정도만 했어요
- 산책을 나가자마자 걷기 싫어했어요
- 하루 종일 누워 있었어요
- 계단을 오르기 싫어했어요

#### symptoms

- 구토나 설사는 없었어요
- 오후에 노란 토를 한 번 했어요
- 묽은 변을 두 번 봤어요
- 기운이 없어 보였어요
- 오른쪽 뒷다리를 살짝 절뚝거렸어요

#### stool

- 정상 변으로 한 번 봤어요
- 묽은 변을 두 번 봤어요
- 아직 배변을 하지 않았어요
- 배변은 평소와 비슷했어요

#### notes

- 평소보다 잠을 많이 잤어요
- 만지면 싫어했어요
- 장난감에 관심이 없었어요
- 특별히 아파 보이는 모습은 없었어요

## 4. 오늘의 상태 기록 샘플

### normal_daily_log

```yaml
sample_id: normal_daily_log
type: daily_log
risk_hint: normal
input_text: |
  오늘은 사료를 평소처럼 거의 다 먹었어요.
  물도 평소랑 비슷하게 마셨고, 산책은 아침에 25분 정도 했어요.
  배변은 정상 변으로 한 번 했고, 구토나 설사는 없었어요.
  특별히 아파 보이는 모습은 없었습니다.
expected_fields:
  food: 평소처럼 섭취
  water: 평소와 비슷함
  activity: 산책 25분
  symptoms: 없음
  stool: 정상 변 1회
```

### reduced_food_daily_log

```yaml
sample_id: reduced_food_daily_log
type: daily_log
risk_hint: observe
input_text: |
  아침 사료를 절반 정도 남겼고 저녁도 평소보다 적게 먹었어요.
  물은 평소처럼 마신 것 같고, 산책은 15분 정도만 했어요.
  평소보다 걷기 싫어하는 느낌이 있었고 기운이 조금 없어 보였어요.
  배변은 정상으로 한 번 했고 구토는 없었어요.
expected_fields:
  food: 평소보다 감소
  water: 평소와 비슷함
  activity: 산책 15분, 평소보다 적음
  symptoms: 기력 저하
  stool: 정상 변 1회
```

### vomiting_daily_log

```yaml
sample_id: vomiting_daily_log
type: daily_log
risk_hint: consult_possible
input_text: |
  아침에는 밥을 조금 남겼고 오후에 노란 토를 한 번 했어요.
  그 이후에는 간식을 조금 먹었고 물은 평소처럼 마셨어요.
  산책은 하지 않았고 집에서 계속 쉬었어요.
  설사는 없었고 배변은 아직 하지 않았어요.
expected_fields:
  food: 감소
  water: 평소와 비슷함
  activity: 산책 없음
  symptoms: 노란 구토 1회
  stool: 배변 없음
```

### diarrhea_daily_log

```yaml
sample_id: diarrhea_daily_log
type: daily_log
risk_hint: observe
input_text: |
  오늘 사료는 평소보다 조금 적게 먹었어요.
  물은 많이 마신 것 같고, 오후에 묽은 변을 두 번 봤어요.
  구토는 없었지만 배에서 소리가 조금 났고 평소보다 조용했어요.
  산책은 10분 정도만 했습니다.
expected_fields:
  food: 약간 감소
  water: 증가 가능
  activity: 산책 10분
  symptoms: 묽은 변 2회
  stool: 묽은 변
```

### reduced_activity_daily_log

```yaml
sample_id: reduced_activity_daily_log
type: daily_log
risk_hint: observe
input_text: |
  밥과 물은 평소와 비슷했어요.
  그런데 산책을 나가자마자 걷기 싫어했고 5분 만에 들어왔어요.
  오른쪽 뒷다리를 살짝 드는 것처럼 보였고 계단을 오르기 싫어했어요.
  구토나 설사는 없었고 배변도 정상이었어요.
expected_fields:
  food: 평소와 비슷함
  water: 평소와 비슷함
  activity: 산책 5분, 감소
  symptoms: 오른쪽 뒷다리 불편
  stool: 정상
```

## 5. AI 상태 체크 입력 구조

### 필드

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| concern | 보호자의 걱정 | 밥을 거의 안 먹어요 |
| onset | 시작 시점 | 오늘부터, 3일 전부터 |
| duration | 지속 기간 | 하루 종일, 3일째 |
| severity | 심각도 표현 | 조금, 많이, 거의 못함 |
| related_symptoms | 동반 증상 | 구토, 설사, 무기력 |
| known_context | 보호자가 알고 있는 맥락 | 기존 질환, 복용약 |
| unknowns | 모르는 정보 | 정확한 음수량은 모름 |

### 자연어 템플릿

```text
{name}가 {concern}.
{onset}부터 {duration} 이어지고 있어요.
같이 보이는 증상은 {related_symptoms}.
{known_context}
{unknowns}
```

## 6. AI 상태 체크 샘플

### check_reduced_food

```yaml
sample_id: check_reduced_food
type: health_check
risk_hint: consult_possible
input_text: |
  코코가 오늘 밥을 거의 안 먹었어요.
  어제도 평소보다 적게 먹었고 오늘은 간식도 별로 관심이 없어요.
  기운도 조금 없어 보여요.
tags:
  - food_decrease
  - lethargy
```

### check_vomiting

```yaml
sample_id: check_vomiting
type: health_check
risk_hint: observe_or_consult
input_text: |
  오후에 노란 토를 한 번 했어요.
  아침밥은 조금 남겼고 지금은 누워서 쉬고 있어요.
  물은 조금 마셨고 설사는 없어요.
tags:
  - vomiting
  - food_decrease
```

### check_diarrhea

```yaml
sample_id: check_diarrhea
type: health_check
risk_hint: observe
input_text: |
  오늘 묽은 변을 세 번 봤어요.
  밥은 조금 먹었고 물은 평소보다 많이 마신 것 같아요.
  구토는 없지만 배가 불편해 보입니다.
tags:
  - diarrhea
  - water_increase
```

### check_limping

```yaml
sample_id: check_limping
type: health_check
risk_hint: consult_possible
input_text: |
  오른쪽 뒷다리를 살짝 절뚝거리는 것 같아요.
  계단을 오르려고 하지 않고 안아달라고 해요.
  밥과 물은 평소와 비슷하고 구토나 설사는 없어요.
tags:
  - limping
  - activity_decrease
```

### check_respiratory_emergency

```yaml
sample_id: check_respiratory_emergency
type: health_check
risk_hint: emergency
input_text: |
  숨을 너무 가쁘게 쉬고 잇몸이 파래 보여요.
  몸을 잘 못 가누고 축 처져 있어요.
tags:
  - respiratory_distress
  - cyanosis_suspected
  - lethargy
```

### check_poisoning_suspected

```yaml
sample_id: check_poisoning_suspected
type: health_check
risk_hint: emergency_possible
input_text: |
  산책 중에 뭔가를 주워 먹은 것 같아요.
  그 뒤로 침을 많이 흘리고 구역질을 해요.
  무엇을 먹었는지는 정확히 못 봤어요.
tags:
  - toxin_exposure_possible
  - drooling
  - retching
  - unknown_ingestion
```

## 7. AI 추가 질문 답변 구조

추가 질문 답변은 짧아도 되지만, 횟수, 시점, 색, 양, 지속 시간을 포함하면 샘플 품질이 좋아진다.

### 답변 필드

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| target_topic | 질문 주제 | 구토, 설사, 식사, 음수 |
| count | 횟수 | 1회, 2회, 모름 |
| time | 발생 시점 | 오후 2시쯤, 새벽 |
| detail | 세부 묘사 | 노란색, 거품, 묽은 변 |
| uncertainty | 불확실한 점 | 정확한 양은 모름 |

### 답변 샘플

```yaml
sample_id: followup_vomiting_yellow_once
type: followup_answer
target_topic: vomiting
input_text: |
  오후에 한 번 노란색 토를 했어요.
  음식물은 거의 없었고 거품이 조금 있었어요.
```

```yaml
sample_id: followup_diarrhea_twice
type: followup_answer
target_topic: diarrhea
input_text: |
  묽은 변을 오후에 두 번 봤어요.
  피는 보이지 않았고 냄새가 평소보다 심했어요.
```

```yaml
sample_id: followup_water_unknown_more
type: followup_answer
target_topic: water
input_text: |
  정확한 양은 모르겠지만 평소보다 물그릇이 더 빨리 비었어요.
```

```yaml
sample_id: followup_existing_condition
type: followup_answer
target_topic: medical_history
input_text: |
  슬개골 탈구 2기 진단을 받았고 관절 영양제를 하루 한 번 먹고 있어요.
  처방약은 지금은 따로 없어요.
```

## 8. 병원 전달용 요약 보완 입력 구조

병원 전달용 요약을 만들기 전, 사용자가 부족한 정보를 직접 보완할 수 있다.

### 보완 필드

| 필드 | 설명 | 예시 |
| --- | --- | --- |
| onset_detail | 발생 시점 | 3일 전부터 식사량 감소 |
| owner_observation | 보호자 관찰 | 산책을 거부함 |
| key_message_for_vet | 병원에 말하고 싶은 내용 | 다리 상태도 같이 확인받고 싶음 |
| medication_note | 복용 정보 | 관절 영양제 1일 1회 |
| allergy_note | 알레르기 | 닭고기 알레르기 |

### 보완 입력 샘플

```yaml
sample_id: handoff_onset_food_vomit
type: handoff_input
input_text: |
  식사량이 줄어든 건 3일 전부터였고, 구토는 오늘 오후 2시쯤 한 번 있었어요.
```

```yaml
sample_id: handoff_owner_observation_activity
type: handoff_input
input_text: |
  평소에는 산책하자고 하면 바로 뛰어나오는데 오늘은 현관 앞에서 움직이지 않으려고 했어요.
  잠도 평소보다 많이 잤습니다.
```

```yaml
sample_id: handoff_vet_message_weight_leg
type: handoff_input
input_text: |
  최근 2주 사이 몸무게가 조금 줄었고, 슬개골 탈구 진단을 받은 적이 있어서 다리 상태도 같이 확인받고 싶어요.
```

## 9. 데모 케이스 조합 규칙

아래 조합을 사용하면 다양한 데모 샘플을 빠르게 만들 수 있다.

| 케이스 | 프로필 | 오늘의 기록 | 상태 체크 | 추가 답변 | 기대 흐름 |
| --- | --- | --- | --- | --- | --- |
| 정상 관찰 | dog_001 | normal_daily_log | 없음 | 없음 | 기록 저장 |
| 관찰 필요 | dog_001 | reduced_food_daily_log | check_reduced_food | followup_water_unknown_more | 관찰 또는 상담 |
| 구토 상담 | dog_001 | vomiting_daily_log | check_vomiting | followup_vomiting_yellow_once | 상담 권장 가능 |
| 설사 관찰 | cat_001 | diarrhea_daily_log | check_diarrhea | followup_diarrhea_twice | 관찰 또는 상담 |
| 다리 불편 | dog_001 | reduced_activity_daily_log | check_limping | followup_existing_condition | 상담 권장 가능 |
| 응급 호흡 | dog_001 | 없음 | check_respiratory_emergency | 없음 | 즉시 병원 안내 |
| 중독 의심 | dog_001 | 없음 | check_poisoning_suspected | 없음 | 응급 또는 빠른 상담 |

## 10. 새 샘플 작성 템플릿

새 샘플을 추가할 때는 아래 형식을 사용한다.

```yaml
sample_id: short_unique_id
type: daily_log | health_check | followup_answer | handoff_input
risk_hint: normal | observe | consult_possible | emergency_possible | emergency
pet_profile: dog_001
input_text: |
  사용자가 실제로 입력할 자연어 문장.
tags:
  - food_decrease
  - vomiting
expected_fields:
  food: optional
  water: optional
  activity: optional
  symptoms: optional
  stool: optional
notes:
  - 정확하지 않은 정보는 추정하지 않는다.
  - 응급 표현은 Safety Agent 테스트에 사용한다.
```

## 11. 태그 목록

샘플 검색과 테스트 분류를 쉽게 하기 위해 아래 태그를 사용한다.

### 식사

- food_normal
- food_decrease
- food_refusal

### 음수

- water_normal
- water_increase
- water_decrease
- water_unknown

### 활동

- activity_normal
- activity_decrease
- walk_refusal
- limping

### 증상

- vomiting
- yellow_vomit
- diarrhea
- lethargy
- respiratory_distress
- cyanosis_suspected
- toxin_exposure_possible
- drooling
- retching

### 배변과 배뇨

- stool_normal
- loose_stool
- no_stool
- urination_decrease
- dark_urine

### 안전 단계

- risk_normal
- risk_observe
- risk_consult_possible
- risk_emergency_possible
- risk_emergency

## 12. 입력 작성 기준

좋은 사용자 입력 샘플은 다음 정보를 포함한다.

- 언제부터 증상이 있었는지
- 몇 번 발생했는지
- 평소와 비교해 얼마나 달라졌는지
- 식사, 음수, 활동, 배변, 구토 여부
- 기존 질환이나 복용 중인 약
- 보호자가 직접 본 중요한 장면

사용자가 정확히 모르는 정보는 억지로 추정하지 않는다.

```text
정확한 양은 모르겠지만 평소보다 적게 마신 것 같아요.
```

```text
무엇을 먹었는지는 정확히 못 봤어요.
```

"""RAG Query Builder — 명세 12절.

사용자 문장을 그대로 검색어로 쓰지 않는다. Cornell 문서는 **영어**이고 사용자
입력은 **한국어**라서, 원문을 그대로 던지면 임베딩 공간에서 겉돈다. 또 "기침해요"
한 마디만으로는 8살 승모판질환 푸들과 1살 건강한 고양이를 구분할 수 없다.
그래서 임상 context 를 붙여 한국어·영어 query 두 개를 만든다.

**Context 우선순위(명세 12절)**

    1. 현재 사용자 입력   ← 항상 최우선(지금 상태이므로)
    2. PET DB(pet_profile: 종/품종/나이/기존질환/복용약)
    3. 진단서 DB(related_diagnoses)
    4. 일기장 DB(supporting_daily_entries)

같은 증상이 여러 소스에 있으면 **먼저 등장한 소스의 표현을 쓰고 뒤는 버린다.**
과거 기록이 현재 호소를 밀어내면 "지금 무슨 일이 벌어지는가"를 놓치기 때문이다.

**LLM 없이도 반드시 동작한다.** 규칙 기반 경로(증상 키워드 사전 + pet 프로필 조합)가
1차 구현이고, LLM 은 그 결과를 다듬는 옵션이다. LLM 호출이 실패하거나 스키마를
어기면 규칙 기반 결과를 그대로 돌려준다(`petcare_ai.llm.safe_structured_invoke`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config import Species
from ..llm import safe_structured_invoke
from ..schemas import RagQuery

logger = logging.getLogger(__name__)

__all__ = [
    "SYMPTOM_TERMS",
    "SYMPTOM_LEXICON",
    "DISEASE_LEXICON",
    "MEDICATION_LEXICON",
    "BREED_LEXICON",
    "DEFAULT_REQUIRED_TOPICS",
    "EMERGENCY_REQUIRED_TOPICS",
    "SENIOR_AGE_YEARS",
    "SymptomTerm",
    "detect_symptoms",
    "resolve_species",
    "build_rag_query",
    "build_rag_query_rule_based",
]


#: 노령 기준(년). 개·고양이 모두 7살부터 노령성 질환 유병률이 뚜렷이 올라가고,
#: Cornell 문서도 "older dog / senior cat" 표현으로 별도 절을 둔다.
SENIOR_AGE_YEARS: float = 7.0

#: 이 나이 미만이면 성장기로 보고 puppy/kitten 표현을 쓴다(질환 스펙트럼이 다르다).
JUVENILE_AGE_YEARS: float = 1.0

#: 명세 12절 출력 예시의 기본 토픽. 어떤 질문이든 "무엇이 위험 신호인지 /
#: 보호자가 무엇을 관찰해야 하는지 / 언제 응급으로 가야 하는지"는 항상 필요하다.
DEFAULT_REQUIRED_TOPICS: tuple[str, ...] = (
    "red flags",
    "owner observations",
    "when to seek emergency veterinary care",
)

#: 응급 힌트가 잡혔을 때 추가하는 토픽.
EMERGENCY_REQUIRED_TOPICS: tuple[str, ...] = (
    "emergency warning signs",
    "immediate veterinary attention",
)

#: query 에 넣을 증상 개수 상한. 너무 많이 넣으면 임베딩이 평균화돼 오히려
#: 검색이 뭉개진다(핵심 증상 2~3개일 때 가장 잘 맞는다).
MAX_SYMPTOMS_IN_QUERY: int = 3
MAX_DISEASES_IN_QUERY: int = 2
MAX_MEDICATIONS_IN_QUERY: int = 2

#: 사용자 문장을 그대로 쓸 때의 길이 상한(증상 매칭이 하나도 안 됐을 때만 사용).
MAX_RAW_MESSAGE_CHARS: int = 60


# ---------------------------------------------------------------------------
# 증상 사전 (한 ↔ 영)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SymptomTerm:
    """증상 1개의 한국어/영어 표준 표현과 사용자 표현 변형.

    `aliases` 에 한국어 구어체와 영어 표현을 함께 넣는다. 보호자는 "숨쉬기 힘들어
    보여"처럼 쓰지 "호흡곤란"이라고 쓰지 않고, 영어로 입력하는 사용자도 있다.
    """

    key: str
    ko: str
    en: str
    aliases: tuple[str, ...]
    emergency: bool = False
    #: 이 증상이 잡히면 함께 매칭된 더 일반적인 증상을 지운다(중복 표현 방지).
    overrides: tuple[str, ...] = field(default_factory=tuple)


#: 증상 키워드 사전. 순서는 탐지 동점 시의 우선순위로도 쓰인다.
SYMPTOM_TERMS: tuple[SymptomTerm, ...] = (
    SymptomTerm(
        key="dyspnea",
        ko="호흡곤란",
        en="respiratory distress",
        aliases=(
            "호흡곤란", "숨쉬기 힘", "숨 쉬기 힘", "숨쉬는 게 힘", "숨을 헐떡", "헐떡",
            "숨차", "숨이 가쁘", "호흡이 가쁘", "개구호흡", "혀를 내밀고",
            "difficulty breathing", "respiratory distress", "dyspnea", "labored breathing",
        ),
        emergency=True,
    ),
    SymptomTerm(
        key="collapse",
        ko="쓰러짐",
        en="collapse and syncope",
        aliases=("쓰러", "실신", "기절", "의식이 없", "의식을 잃", "collapse", "syncope", "faint"),
        emergency=True,
    ),
    SymptomTerm(
        key="seizure",
        ko="경련",
        en="seizure",
        aliases=("경련", "발작", "seizure", "convuls"),
        emergency=True,
    ),
    SymptomTerm(
        key="paralysis",
        ko="마비",
        en="paralysis and limb weakness",
        aliases=("마비", "뒷다리를 못 쓰", "일어서지 못", "일어나지 못", "paralysis", "paresis"),
        emergency=True,
    ),
    SymptomTerm(
        key="pale_gums",
        ko="잇몸 창백",
        en="pale gums",
        aliases=("잇몸이 창백", "잇몸이 하얗", "창백", "pale gum", "pale mucous"),
        emergency=True,
    ),
    SymptomTerm(
        key="bleeding",
        ko="출혈",
        en="bleeding",
        aliases=("출혈", "피가 나", "피를 흘", "피를 토", "혈뇨", "혈변", "bleeding", "hemorrhage"),
        emergency=True,
    ),
    SymptomTerm(
        key="abdominal_distension",
        ko="복부 팽만",
        en="abdominal distension and bloat",
        aliases=("복부 팽만", "배가 부풀", "배가 빵빵", "복부가 부", "bloat", "abdominal distension", "gdv"),
        emergency=True,
    ),
    SymptomTerm(
        key="urinary_obstruction",
        ko="배뇨곤란",
        en="urinary obstruction and straining to urinate",
        aliases=(
            "소변을 못", "오줌을 못", "배뇨곤란", "소변이 안 나", "소변을 힘들",
            "urinary obstruction", "straining to urinate", "blocked cat",
        ),
        emergency=True,
        overrides=("urinary",),
    ),
    SymptomTerm(
        key="lily_toxicity",
        ko="백합 중독",
        en="lily toxicity",
        aliases=("백합", "나리꽃", "lily"),
        emergency=True,
        overrides=("poisoning",),
    ),
    SymptomTerm(
        key="chocolate_toxicity",
        ko="초콜릿 중독",
        en="chocolate toxicity",
        aliases=("초콜릿", "초콜렛", "chocolate", "theobromine"),
        emergency=True,
        overrides=("poisoning", "foreign_body"),
    ),
    SymptomTerm(
        key="grape_toxicity",
        ko="포도·건포도 중독",
        en="grape and raisin toxicity",
        aliases=("포도", "건포도", "grape", "raisin"),
        emergency=True,
        overrides=("poisoning", "foreign_body"),
    ),
    SymptomTerm(
        key="onion_toxicity",
        ko="양파·마늘 중독",
        en="onion and garlic toxicity",
        aliases=("양파", "마늘", "부추", "onion", "garlic"),
        emergency=True,
        overrides=("poisoning", "foreign_body"),
    ),
    SymptomTerm(
        key="xylitol_toxicity",
        ko="자일리톨 중독",
        en="xylitol toxicity",
        aliases=("자일리톨", "xylitol"),
        emergency=True,
        overrides=("poisoning", "foreign_body"),
    ),
    SymptomTerm(
        key="poisoning",
        ko="중독",
        en="poisoning and toxin exposure",
        aliases=("중독", "독성", "살충제", "쥐약", "세제를 먹", "poison", "toxin", "toxicity"),
        emergency=True,
    ),
    SymptomTerm(
        key="foreign_body",
        ko="이물 섭취",
        en="foreign body ingestion",
        aliases=("이물", "삼켰", "삼켜", "먹어버렸", "먹어 버렸", "foreign body", "ingested"),
        emergency=True,
    ),
    SymptomTerm(
        key="jaundice",
        ko="황달",
        en="jaundice",
        aliases=("황달", "눈이 노랗", "jaundice", "icterus"),
        emergency=True,
    ),
    SymptomTerm(
        key="cough",
        ko="기침",
        en="cough",
        aliases=("기침", "콜록", "cough"),
    ),
    SymptomTerm(
        key="vomiting",
        ko="구토",
        en="vomiting",
        aliases=("구토", "토해", "토하", "토함", "게워", "vomit"),
    ),
    SymptomTerm(
        key="diarrhea",
        ko="설사",
        en="diarrhea",
        aliases=("설사", "묽은 변", "무른 변", "diarrhea", "loose stool"),
    ),
    SymptomTerm(
        key="anorexia",
        ko="식욕부진",
        en="loss of appetite",
        aliases=(
            "식욕부진", "식욕이 없", "식욕 저하", "밥을 안 먹", "사료를 안 먹",
            "잘 안 먹", "입맛이 없", "anorexia", "loss of appetite", "not eating",
        ),
    ),
    SymptomTerm(
        key="lethargy",
        ko="기력저하",
        en="lethargy",
        aliases=(
            "기력저하", "기력이 없", "기운이 없", "무기력", "축 처", "쳐져 있",
            "활력이 없", "lethargy", "lethargic",
        ),
    ),
    SymptomTerm(
        key="urinary",
        ko="배뇨 이상",
        en="abnormal urination",
        aliases=("소변", "오줌", "배뇨", "urination", "urinary"),
    ),
    SymptomTerm(
        key="limping",
        ko="절뚝거림",
        en="limping and lameness",
        aliases=("절뚝", "다리를 절", "파행", "다리를 들고", "limp", "lameness"),
    ),
    SymptomTerm(
        key="fever",
        ko="발열",
        en="fever",
        aliases=("발열", "열이 나", "고열", "체온이 높", "fever", "pyrexia"),
    ),
    SymptomTerm(
        key="itching",
        ko="가려움",
        en="itching and pruritus",
        aliases=("가려", "긁어", "긁는", "핥아", "itch", "pruritus", "scratching"),
    ),
    SymptomTerm(
        key="skin_lesion",
        ko="피부 병변",
        en="skin lesions and rash",
        aliases=("피부", "발진", "각질", "붉어졌", "탈모", "skin lesion", "rash", "hair loss"),
    ),
    SymptomTerm(
        key="ear",
        ko="귀 이상",
        en="ear infection and otitis",
        aliases=("귀를 긁", "귀에서 냄새", "외이염", "귀 분비물", "ear infection", "otitis"),
    ),
    SymptomTerm(
        key="eye",
        ko="눈 이상",
        en="eye discharge and ocular problems",
        aliases=("눈곱", "눈이 충혈", "눈물이 많", "각막", "eye discharge", "ocular", "conjunctiv"),
    ),
    SymptomTerm(
        key="weight_loss",
        ko="체중 감소",
        en="weight loss",
        aliases=("체중 감소", "체중이 줄", "살이 빠", "몸무게가 줄", "weight loss"),
    ),
    SymptomTerm(
        key="drooling",
        ko="침 흘림",
        en="drooling and hypersalivation",
        aliases=("침을 흘", "침 흘", "침이 많", "drool", "hypersalivation", "ptyalism"),
    ),
    SymptomTerm(
        key="dehydration",
        ko="탈수",
        en="dehydration",
        aliases=("탈수", "dehydrat"),
    ),
    SymptomTerm(
        key="polydipsia",
        ko="다음다뇨",
        en="increased thirst and urination",
        aliases=("물을 많이", "물을 자주", "다음다뇨", "polydipsia", "polyuria", "increased thirst"),
        overrides=("urinary",),
    ),
    SymptomTerm(
        key="constipation",
        ko="변비",
        en="constipation",
        # "변을 못" 은 "소변을 못" 에도 걸려 배뇨곤란을 변비로 오인하므로 쓰지 않는다.
        aliases=("변비", "대변을 못", "똥을 못", "응가를 못", "constipation"),
    ),
    SymptomTerm(
        key="sneezing",
        ko="재채기",
        en="sneezing and nasal discharge",
        aliases=("재채기", "콧물", "코를 훌쩍", "sneez", "nasal discharge"),
    ),
    SymptomTerm(
        key="tremor",
        ko="떨림",
        en="tremors",
        aliases=("떨림", "떨어요", "떨고 있", "몸을 떨", "tremor", "shaking"),
    ),
    SymptomTerm(
        key="dental",
        ko="구취·치아 문제",
        en="bad breath and dental disease",
        aliases=("구취", "입냄새", "입 냄새", "치석", "이가 흔들", "dental", "bad breath", "periodont"),
    ),
    SymptomTerm(
        key="behavior_change",
        ko="행동 변화",
        en="behavioral changes",
        aliases=("행동이 이상", "성격이 변", "숨어 있", "안 놀", "behavior change", "behaviour change"),
    ),
)

#: key → SymptomTerm 조회용.
SYMPTOM_LEXICON: dict[str, SymptomTerm] = {term.key: term for term in SYMPTOM_TERMS}


# ---------------------------------------------------------------------------
# 질환 / 약물 / 품종 사전
# ---------------------------------------------------------------------------
#: (매칭 키워드들, 한국어 표현, 영어 표현). 진단서·PET DB 의 질환명은 한국어일 수도
#: 영어일 수도 있어서 양쪽 키워드를 함께 넣고 부분 일치로 찾는다.
DISEASE_LEXICON: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("mitral", "heart", "cardiac", "cardio", "chf", "심장", "심부전", "승모판"), "심장질환", "heart disease"),
    (("kidney", "renal", "ckd", "신장", "신부전"), "신장질환", "kidney disease"),
    (("diabet", "당뇨"), "당뇨병", "diabetes mellitus"),
    (("pancreat", "췌장"), "췌장염", "pancreatitis"),
    (("liver", "hepat", "간질환", "간 질환"), "간질환", "liver disease"),
    (("epilep", "뇌전증", "간질"), "뇌전증", "epilepsy"),
    (("arthrit", "관절", "슬개골", "patell"), "관절질환", "arthritis and joint disease"),
    (("asthma", "천식"), "천식", "asthma"),
    (("hyperthyroid", "갑상선기능항진"), "갑상선기능항진증", "hyperthyroidism"),
    (("hypothyroid", "갑상선기능저하"), "갑상선기능저하증", "hypothyroidism"),
    (("cushing", "쿠싱", "hyperadrenocortic"), "부신피질기능항진증", "Cushing's disease"),
    (("inflammatory bowel", "ibd", "염증성 장"), "염증성 장질환", "inflammatory bowel disease"),
    (("allerg", "atop", "알레르기", "아토피"), "알레르기 피부질환", "allergic skin disease"),
    (("cancer", "tumor", "tumour", "lymphoma", "종양", "림프종", "암"), "종양", "cancer"),
    (("obes", "비만"), "비만", "obesity"),
    (("flutd", "cystitis", "bladder", "방광", "요로"), "하부요로질환", "lower urinary tract disease"),
    (("periodont", "dental", "치주", "치과"), "치주질환", "periodontal disease"),
    (("collaps", "tracheal", "기관허탈", "기관 협착"), "기관허탈", "tracheal collapse"),
    (("felv", "fiv", "고양이 백혈병", "면역결핍"), "고양이 레트로바이러스 감염", "feline retrovirus infection"),
    (("pneumon", "폐렴"), "폐렴", "pneumonia"),
)

#: 복용약 사전. 약은 "무슨 병을 관리 중인가"의 단서라서 query context 에 넣는다.
MEDICATION_LEXICON: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("heart", "cardiac", "pimobendan", "심장약", "심장"), "심장약", "heart medication"),
    (("furosemide", "diuretic", "이뇨"), "이뇨제", "diuretic"),
    (("insulin", "인슐린"), "인슐린", "insulin"),
    (("prednis", "steroid", "스테로이드"), "스테로이드", "corticosteroid"),
    (("antibiot", "항생"), "항생제", "antibiotic"),
    (("nsaid", "painkill", "진통", "소염진통"), "진통제", "pain medication"),
    (("phenobarb", "levetiracetam", "항경련", "발작약"), "항경련제", "anti-seizure medication"),
    (("thyroid", "갑상선"), "갑상선약", "thyroid medication"),
    (("antihistamin", "항히스타민"), "항히스타민제", "antihistamine"),
)

#: 품종 한↔영. 앱 입력이 영어 소문자라는 보장이 없어서 양방향으로 찾는다.
BREED_LEXICON: dict[str, tuple[str, str]] = {
    "poodle": ("푸들", "poodle"),
    "푸들": ("푸들", "poodle"),
    "maltese": ("말티즈", "Maltese"),
    "말티즈": ("말티즈", "Maltese"),
    "pomeranian": ("포메라니안", "Pomeranian"),
    "포메라니안": ("포메라니안", "Pomeranian"),
    "chihuahua": ("치와와", "Chihuahua"),
    "치와와": ("치와와", "Chihuahua"),
    "shih tzu": ("시츄", "Shih Tzu"),
    "시츄": ("시츄", "Shih Tzu"),
    "yorkshire terrier": ("요크셔테리어", "Yorkshire Terrier"),
    "요크셔테리어": ("요크셔테리어", "Yorkshire Terrier"),
    "welsh corgi": ("웰시코기", "Welsh Corgi"),
    "corgi": ("웰시코기", "Welsh Corgi"),
    "웰시코기": ("웰시코기", "Welsh Corgi"),
    "golden retriever": ("골든리트리버", "Golden Retriever"),
    "골든리트리버": ("골든리트리버", "Golden Retriever"),
    "labrador": ("래브라도리트리버", "Labrador Retriever"),
    "래브라도": ("래브라도리트리버", "Labrador Retriever"),
    "beagle": ("비글", "Beagle"),
    "비글": ("비글", "Beagle"),
    "bichon": ("비숑프리제", "Bichon Frise"),
    "비숑": ("비숑프리제", "Bichon Frise"),
    "dachshund": ("닥스훈트", "Dachshund"),
    "닥스훈트": ("닥스훈트", "Dachshund"),
    "french bulldog": ("프렌치불독", "French Bulldog"),
    "프렌치불독": ("프렌치불독", "French Bulldog"),
    "pug": ("퍼그", "Pug"),
    "퍼그": ("퍼그", "Pug"),
    "korean short hair": ("코리안숏헤어", "domestic shorthair"),
    "코리안숏헤어": ("코리안숏헤어", "domestic shorthair"),
    "코숏": ("코리안숏헤어", "domestic shorthair"),
    "persian": ("페르시안", "Persian"),
    "페르시안": ("페르시안", "Persian"),
    "russian blue": ("러시안블루", "Russian Blue"),
    "러시안블루": ("러시안블루", "Russian Blue"),
    "scottish fold": ("스코티시폴드", "Scottish Fold"),
    "스코티시폴드": ("스코티시폴드", "Scottish Fold"),
    "siamese": ("샴", "Siamese"),
    "샴": ("샴", "Siamese"),
    "ragdoll": ("랙돌", "Ragdoll"),
    "랙돌": ("랙돌", "Ragdoll"),
    "maine coon": ("메인쿤", "Maine Coon"),
    "메인쿤": ("메인쿤", "Maine Coon"),
    "abyssinian": ("아비시니안", "Abyssinian"),
    "bengal": ("벵갈", "Bengal"),
    "벵갈": ("벵갈", "Bengal"),
}

#: "지금 응급인가"를 문장에서 직접 잡는 신호. 증상 사전과 별개로 둔다.
EMERGENCY_PHRASES: tuple[str, ...] = (
    "응급", "위급", "죽을 것 같", "죽어가", "숨을 안 쉬", "숨을 못 쉬",
    "심정지", "심폐소생", "emergency", "urgent", "not breathing",
)

#: 진단서/일기장에서 텍스트를 찾을 때 볼 키(앞에 있는 키가 우선).
_DIAGNOSIS_TEXT_KEYS: tuple[str, ...] = (
    "diagnosis", "diagnosis_name", "disease", "disease_name", "condition",
    "title", "summary", "content", "notes", "note", "text", "description",
)
_DAILY_TEXT_KEYS: tuple[str, ...] = (
    "symptoms", "symptom", "content", "text", "note", "notes", "memo", "title", "summary",
)

_SPECIES_ALIASES: dict[str, Species] = {
    "dog": "dog", "canine": "dog", "puppy": "dog",
    "강아지": "dog", "개": "dog", "견": "dog", "반려견": "dog",
    "cat": "cat", "feline": "cat", "kitten": "cat",
    "고양이": "cat", "묘": "cat", "반려묘": "cat",
}

#: 약물/질환 표현이 겹치는지 볼 때 무시할 일반 명사.
_GENERIC_TOKENS: frozenset[str] = frozenset(
    {"medication", "medicine", "drug", "disease", "and", "of", "the", "a", "an"}
)


# ---------------------------------------------------------------------------
# 한국어 조사/문자열 헬퍼
# ---------------------------------------------------------------------------
def _has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있는지 — "기침과 / 호흡곤란과" 같은 조사 선택에 쓴다.

    한글 음절은 (코드 - 0xAC00) % 28 이 0 이 아니면 종성이 있다.
    한글이 아니면(영문·숫자) 받침 없음으로 본다.
    """
    if not word:
        return False
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _join_ko(items: Sequence[str]) -> str:
    """한국어 나열 — 마지막 항목만 "과/와"로 잇는다("기침과 호흡곤란")."""
    parts = [item for item in items if item]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    head = ", ".join(parts[:-1])
    particle = "과" if _has_final_consonant(parts[-2]) else "와"
    return f"{head}{particle} {parts[-1]}"


def _join_en(items: Sequence[str]) -> str:
    """영어 나열 — "a and b" / "a, b and c"."""
    parts = [item for item in items if item]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _article(phrase: str) -> str:
    """부정관사 — 모음으로 시작하면 an("an older dog")."""
    stripped = phrase.strip().lower()
    return "an" if stripped[:1] in {"a", "e", "i", "o", "u"} else "a"


def _clean_text(value: Any) -> str:
    """어떤 값이든 공백이 정리된 1줄 문자열로 만든다."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_clean_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_clean_text(item) for item in value.values())
    return re.sub(r"\s+", " ", str(value)).strip()


def _dedupe(items: Iterable[str]) -> list[str]:
    """순서를 지키며 중복을 제거한다(우선순위가 순서로 표현되기 때문)."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _as_list(value: Any) -> list[Any]:
    """단일 값/None/리스트를 모두 리스트로 정규화한다(앱 데이터가 들쭉날쭉하다)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value]
    return [value]


# ---------------------------------------------------------------------------
# 탐지 함수
# ---------------------------------------------------------------------------
def detect_symptoms(text: str) -> list[SymptomTerm]:
    """문장에서 증상을 찾아 **등장 순서대로** 돌려준다.

    한국어는 어절 분리가 어려워 형태소 분석 대신 부분 문자열 매칭을 쓴다.
    동일 위치에서 여러 증상이 걸리면 응급 증상을 먼저 둔다 — "소변을 못 봐요"를
    일반 '배뇨 이상'이 아니라 '배뇨곤란(요도폐색)'으로 읽어야 안전한 쪽으로 틀린다.
    """
    haystack = _clean_text(text).lower()
    if not haystack:
        return []

    hits: list[tuple[int, int, int, SymptomTerm]] = []
    for order, term in enumerate(SYMPTOM_TERMS):
        position = min(
            (haystack.find(alias.lower()) for alias in term.aliases if alias.lower() in haystack),
            default=-1,
        )
        if position >= 0:
            hits.append((position, 0 if term.emergency else 1, order, term))

    hits.sort(key=lambda item: (item[0], item[1], item[2]))
    matched = [item[3] for item in hits]

    # 더 구체적인 증상이 잡혔으면 포괄적인 증상은 지운다(중복 표현 방지).
    superseded = {key for term in matched for key in term.overrides}
    return [term for term in matched if term.key not in superseded]


def resolve_species(pet_profile: dict | None) -> Species:
    """pet_profile 에서 species 를 결정한다 — 값이 없으면 'dog'.

    species 는 검색할 index 를 고르는 값이라 절대 비워 둘 수 없다. 알 수 없을 때
    'dog' 로 두는 이유는 Cornell 코퍼스의 dog 문서가 더 많고(160 vs 123),
    임의로 검색을 실패시키는 것보다 낫기 때문이다. 잘못된 종이 들어오면 경고를 남긴다.
    """
    raw = _clean_text((pet_profile or {}).get("species")).lower()
    if not raw:
        return "dog"
    for alias, species in _SPECIES_ALIASES.items():
        if alias in raw:
            return species
    logger.warning("알 수 없는 species 값 %r — 기본값 'dog' 로 검색합니다.", raw)
    return "dog"


def _resolve_age(pet_profile: dict) -> float | None:
    """나이(년)를 읽는다 — 앱 필드명이 여러 개라 순서대로 시도한다."""
    for key in ("age_years", "age", "나이", "age_year"):
        if key in pet_profile:
            try:
                value = float(pet_profile[key])
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
    # 개월 수만 있는 경우도 흔하다.
    for key in ("age_months", "months"):
        if key in pet_profile:
            try:
                return float(pet_profile[key]) / 12.0
            except (TypeError, ValueError):
                continue
    return None


def _lookup(lexicon: tuple[tuple[tuple[str, ...], str, str], ...], raw: str) -> tuple[str, str] | None:
    """사전에서 부분 일치로 (한국어, 영어) 표준 표현을 찾는다."""
    text = raw.lower()
    for keywords, ko, en in lexicon:
        if any(keyword in text for keyword in keywords):
            return ko, en
    return None


def _resolve_breed(pet_profile: dict) -> tuple[str, str] | None:
    """품종을 (한국어, 영어)로 정규화한다 — 사전에 없으면 원문을 그대로 쓴다."""
    raw = _clean_text(pet_profile.get("breed") or pet_profile.get("품종"))
    if not raw:
        return None
    lowered = raw.lower()
    for alias, pair in BREED_LEXICON.items():
        if alias in lowered:
            return pair
    return raw, raw


def _resolve_diseases(pet_profile: dict, related_diagnoses: list[dict]) -> list[tuple[str, str]]:
    """기존질환 목록을 (한국어, 영어)로 정규화한다.

    PET DB 의 `diseases` 가 진단서보다 우선한다(명세 12절의 DB 우선순위).
    진단서는 PET DB 에 없는 질환을 보충하는 용도로만 뒤에 붙인다.
    """
    raw_terms: list[str] = []
    for key in ("diseases", "disease", "chronic_conditions", "conditions", "기존질환"):
        raw_terms.extend(_clean_text(item) for item in _as_list(pet_profile.get(key)))
    for entry in related_diagnoses:
        if isinstance(entry, dict):
            raw_terms.append(_first_text(entry, _DIAGNOSIS_TEXT_KEYS))
        else:
            raw_terms.append(_clean_text(entry))

    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in raw_terms:
        if not raw:
            continue
        pair = _lookup(DISEASE_LEXICON, raw)
        if pair is None:
            # 사전에 없는 질환명은 버리지 않고 원문을 그대로 쓴다 — 정보 손실 방지.
            pair = (raw, raw)
        if pair[1].lower() in seen:
            continue
        seen.add(pair[1].lower())
        resolved.append(pair)
    return resolved


def _resolve_medications(pet_profile: dict, diseases: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """복용약을 정규화하되 **질환과 의미가 겹치는 약은 뺀다.**

    "심장질환 + 심장약"처럼 같은 단어를 두 번 넣으면 임베딩상 새 정보가 없고
    query 만 길어진다. 질환 영어 표현과 토큰이 겹치면 생략한다.
    """
    disease_tokens = {
        token
        for _, en in diseases
        for token in re.split(r"[^a-z가-힣]+", en.lower())
        if token and token not in _GENERIC_TOKENS
    }

    raw_terms: list[str] = []
    for key in ("medications", "medication", "medicines", "drugs", "복용약"):
        raw_terms.extend(_clean_text(item) for item in _as_list(pet_profile.get(key)))

    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in raw_terms:
        if not raw:
            continue
        pair = _lookup(MEDICATION_LEXICON, raw) or (raw, raw)
        tokens = {
            token
            for token in re.split(r"[^a-z가-힣]+", pair[1].lower())
            if token and token not in _GENERIC_TOKENS
        }
        if tokens & disease_tokens:
            continue
        if pair[1].lower() in seen:
            continue
        seen.add(pair[1].lower())
        resolved.append(pair)
    return resolved


def _first_text(entry: dict, keys: Sequence[str]) -> str:
    """dict 에서 앞선 키부터 찾아 첫 번째 비어 있지 않은 텍스트를 돌려준다."""
    for key in keys:
        value = _clean_text(entry.get(key))
        if value:
            return value
    return ""


def _context_text(entries: list[dict], keys: Sequence[str]) -> str:
    """진단서/일기장 리스트를 증상 탐지용 한 덩어리 텍스트로 만든다."""
    parts: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            parts.extend(_clean_text(entry.get(key)) for key in keys)
        else:
            parts.append(_clean_text(entry))
    return " ".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# 규칙 기반 query 생성
# ---------------------------------------------------------------------------
def _subject_ko(species: Species, age: float | None, breed_ko: str | None) -> str:
    """한국어 주어 — "노령견" / "어린 고양이" / "푸들 강아지"."""
    if species == "cat":
        base = "고양이"
        senior, juvenile = "노령묘", "어린 고양이"
    else:
        base = "강아지"
        senior, juvenile = "노령견", "어린 강아지"

    if age is not None and age >= SENIOR_AGE_YEARS:
        core = senior
    elif age is not None and age < JUVENILE_AGE_YEARS:
        core = juvenile
    else:
        core = base
    return f"{breed_ko} {core}" if breed_ko else core


def _subject_en(species: Species, age: float | None, breed_en: str | None) -> str:
    """영어 주어 — "older dog" / "kitten" / "older Poodle dog".

    Cornell 문서가 실제로 쓰는 표현("older dog", "senior cat", "puppy", "kitten")에
    맞춰 나이 수식어를 고른다. 종 명사(dog/cat)는 품종이 있어도 남긴다 — 검색
    대상 코퍼스가 종 단위로 서술돼 있어 종 단어가 가장 강한 신호이기 때문이다.
    """
    juvenile = age is not None and age < JUVENILE_AGE_YEARS
    senior = age is not None and age >= SENIOR_AGE_YEARS

    parts: list[str] = []
    if senior:
        parts.append("older")
    if breed_en:
        parts.append(breed_en)
    if juvenile:
        parts.append("kitten" if species == "cat" else "puppy")
    else:
        parts.append(species)
    return " ".join(parts)


def build_rag_query_rule_based(
    user_message: str,
    pet_profile: dict,
    related_diagnoses: list[dict],
    supporting_daily_entries: list[dict],
) -> RagQuery:
    """LLM 없이 규칙만으로 `RagQuery` 를 만든다 — 오프라인 기본 경로.

    조합 순서가 곧 명세 12절의 우선순위다.
      1) 현재 사용자 입력에서 증상을 뽑는다(가장 앞에 배치).
      2) 부족하면 진단서 → 일기장 순으로 증상을 보충한다.
      3) PET DB 의 나이/품종/기존질환/복용약으로 주어와 수식어를 만든다.

    `emergency_hint` 는 **현재 사용자 입력에서만** 판정한다. 과거 진단서에 응급
    이력이 있다고 지금을 응급으로 단정하면 오탐이 쌓이고, 반대로 현재 문장의
    응급 신호는 절대 놓치면 안 되기 때문이다.
    """
    profile = pet_profile or {}
    diagnoses = [entry for entry in (related_diagnoses or [])]
    daily_entries = [entry for entry in (supporting_daily_entries or [])]

    species = resolve_species(profile)
    age = _resolve_age(profile)
    message = _clean_text(user_message)

    # --- 1) 증상: 현재 입력 > 진단서 > 일기장 -----------------------------
    current_symptoms = detect_symptoms(message)
    context_symptoms = detect_symptoms(_context_text(diagnoses, _DIAGNOSIS_TEXT_KEYS))
    diary_symptoms = detect_symptoms(_context_text(daily_entries, _DAILY_TEXT_KEYS))

    ordered: list[SymptomTerm] = []
    seen_keys: set[str] = set()
    for term in [*current_symptoms, *context_symptoms, *diary_symptoms]:
        if term.key not in seen_keys:
            seen_keys.add(term.key)
            ordered.append(term)
    symptoms = ordered[:MAX_SYMPTOMS_IN_QUERY]

    # --- 2) PET DB 수식어 --------------------------------------------------
    diseases = _resolve_diseases(profile, diagnoses)[:MAX_DISEASES_IN_QUERY]
    medications = _resolve_medications(profile, diseases)[:MAX_MEDICATIONS_IN_QUERY]
    breed = _resolve_breed(profile)
    # 품종은 기존질환이 없을 때만 넣는다. 질환이 있으면 그쪽이 훨씬 강한 검색 신호이고,
    # 품종까지 붙이면 query 가 길어져 핵심 증상의 비중이 희석된다.
    breed_ko, breed_en = (breed if breed and not diseases else (None, None))

    subject_ko = _subject_ko(species, age, breed_ko)
    subject_en = _subject_en(species, age, breed_en)

    # --- 3) 응급 판정 (현재 입력 기준) ------------------------------------
    lowered_message = message.lower()
    emergency_hint = any(term.emergency for term in current_symptoms) or any(
        phrase in lowered_message for phrase in EMERGENCY_PHRASES
    )

    # --- 4) query 문자열 조립 ---------------------------------------------
    modifier_ko = _build_modifier_ko(diseases, medications)
    modifier_en = _build_modifier_en(diseases, medications)

    if symptoms:
        symptoms_ko = _join_ko([term.ko for term in symptoms])
        symptoms_en = _join_en([term.en for term in symptoms])
        primary_query_ko = f"{modifier_ko}{subject_ko}의 {symptoms_ko} 경고 신호"
        primary_query_en = (
            f"warning signs of {symptoms_en} in {_article(subject_en)} {subject_en}{modifier_en}"
        )
    else:
        # 증상 사전에 걸린 게 없으면 사용자 문장을 그대로 쓴다(정보를 지우지 않는다).
        # 다만 영어 query 는 임의 번역이 불가능하므로 프로필 기반 일반 질의로 만든다.
        topic_ko = message[:MAX_RAW_MESSAGE_CHARS] if message else "건강 이상"
        primary_query_ko = f"{modifier_ko}{subject_ko}의 {topic_ko} 관련 경고 신호와 보호자 관찰 사항"
        primary_query_en = (
            f"warning signs and owner observations for {_article(subject_en)} {subject_en}{modifier_en}"
        )

    return RagQuery(
        primary_query_ko=primary_query_ko,
        primary_query_en=primary_query_en,
        required_topics=_build_required_topics(emergency_hint),
        species=species,
        emergency_hint=emergency_hint,
    )


def _build_modifier_ko(
    diseases: list[tuple[str, str]], medications: list[tuple[str, str]]
) -> str:
    """한국어 수식절 — "심장질환이 있는 " / "심장질환이 있고 인슐린을 복용 중인 "."""
    clauses: list[str] = []
    if diseases:
        names = _join_ko([ko for ko, _ in diseases])
        clauses.append(f"{names}이 있" if _has_final_consonant(names) else f"{names}가 있")
    if medications:
        names = _join_ko([ko for ko, _ in medications])
        clauses.append(f"{names}을 복용 중인" if _has_final_consonant(names) else f"{names}를 복용 중인")

    if not clauses:
        return ""
    if len(clauses) == 1:
        clause = clauses[0]
        return f"{clause}는 " if clause.endswith("있") else f"{clause} "
    head = clauses[0] + ("고" if clauses[0].endswith("있") else "")
    return f"{head} {clauses[1]} "


def _build_modifier_en(
    diseases: list[tuple[str, str]], medications: list[tuple[str, str]]
) -> str:
    """영어 수식절 — " with heart disease" / " with diabetes receiving insulin"."""
    parts: list[str] = []
    if diseases:
        parts.append(f" with {_join_en([en for _, en in diseases])}")
    if medications:
        parts.append(f" receiving {_join_en([en for _, en in medications])}")
    return "".join(parts)


def _build_required_topics(emergency_hint: bool) -> list[str]:
    """필수 토픽 — 기본 3종(명세 12절)에 응급이면 응급 토픽을 더한다.

    토픽은 충분성 판단(명세 14절)의 coverage 계산 기준이라 무한정 늘리지 않는다.
    """
    topics = list(DEFAULT_REQUIRED_TOPICS)
    if emergency_hint:
        topics.extend(EMERGENCY_REQUIRED_TOPICS)
    return _dedupe(topics)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
_LLM_SYSTEM_PROMPT = (
    "You are a veterinary information retrieval assistant. You rewrite a Korean pet owner's "
    "message into search queries for an English veterinary knowledge base (Cornell Riney Canine "
    "Health Center / Cornell Feline Health Center).\n"
    "Rules:\n"
    "- The owner's CURRENT message outranks every stored record. Never drop a symptom the owner "
    "just reported.\n"
    "- Use the pet profile (age, breed, existing conditions, medications) only as qualifiers.\n"
    "- Produce one natural Korean query and one natural English query describing the clinical "
    "picture and the warning signs to look for.\n"
    "- Never invent symptoms, diagnoses, drug names, or dosages. Never state a diagnosis.\n"
    "- Keep each query under 30 words."
)


def build_rag_query(
    user_message: str,
    pet_profile: dict,
    related_diagnoses: list[dict],
    supporting_daily_entries: list[dict],
    llm: Any | None = None,
) -> RagQuery:
    """임상 context 를 조합해 한국어·영어 검색 query 를 만든다(명세 12절).

    `llm=None` 이면 규칙 기반 결과를 그대로 돌려준다. LLM 이 주어지면 규칙 기반
    결과를 기본값(default)으로 넘겨 structured output 으로 개선을 시도하고,
    호출 실패·스키마 위반이면 기본값으로 되돌아간다(`safe_structured_invoke` 가
    예외를 밖으로 내보내지 않는다).

    LLM 결과라도 다음 두 가지는 규칙 기반 판단이 이긴다.
      - `species`: 검색할 index 를 고르는 값이다. LLM 이 바꾸면 강아지 질문에
        고양이 문서를 붙이는 사고가 나므로 pet_profile 값으로 덮어쓴다.
      - `emergency_hint`: 위험도는 낮은 쪽으로 덮어쓰지 않는다(명세 28절과 같은 원칙).
        규칙이 응급이라고 본 것을 LLM 이 해제할 수 없다.
    """
    fallback = build_rag_query_rule_based(
        user_message, pet_profile, related_diagnoses, supporting_daily_entries
    )
    if llm is None:
        return fallback

    payload = _describe_context(
        user_message, pet_profile, related_diagnoses, supporting_daily_entries, fallback
    )
    messages = [("system", _LLM_SYSTEM_PROMPT), ("human", payload)]
    result = safe_structured_invoke(llm, messages, RagQuery, fallback)

    # LLM 이 query 를 비워 보내면(드물지만 발생한다) 규칙 기반으로 되돌린다.
    if not _clean_text(result.primary_query_ko) or not _clean_text(result.primary_query_en):
        logger.warning("LLM 이 빈 query 를 반환해 규칙 기반 query 를 사용합니다.")
        return fallback

    return RagQuery(
        primary_query_ko=_clean_text(result.primary_query_ko),
        primary_query_en=_clean_text(result.primary_query_en),
        required_topics=_dedupe([*fallback.required_topics, *result.required_topics]),
        species=fallback.species,
        emergency_hint=bool(fallback.emergency_hint or result.emergency_hint),
    )


def _describe_context(
    user_message: str,
    pet_profile: dict,
    related_diagnoses: list[dict],
    supporting_daily_entries: list[dict],
    baseline: RagQuery,
) -> str:
    """LLM 에 넘길 context 를 우선순위가 드러나는 순서로 직렬화한다.

    JSON 을 통째로 넣지 않고 우선순위 라벨을 붙여 나열하는 이유는, 모델이
    "지금 호소 > 과거 기록" 순서를 텍스트 구조에서 바로 읽게 하기 위해서다.
    """
    profile = pet_profile or {}
    lines = [
        "[1. CURRENT OWNER MESSAGE — highest priority]",
        _clean_text(user_message) or "(없음)",
        "",
        "[2. PET DB]",
        f"species={profile.get('species') or '(unknown)'}, "
        f"breed={profile.get('breed') or '(unknown)'}, "
        f"age_years={profile.get('age_years') or profile.get('age') or '(unknown)'}",
        f"diseases={_clean_text(profile.get('diseases')) or '(none)'}",
        f"medications={_clean_text(profile.get('medications')) or '(none)'}",
        "",
        "[3. DIAGNOSIS DB]",
        _context_text(list(related_diagnoses or []), _DIAGNOSIS_TEXT_KEYS) or "(없음)",
        "",
        "[4. DAILY DIARY DB]",
        _context_text(list(supporting_daily_entries or []), _DAILY_TEXT_KEYS) or "(없음)",
        "",
        "[Rule-based baseline — improve on it, do not contradict it]",
        f"primary_query_ko: {baseline.primary_query_ko}",
        f"primary_query_en: {baseline.primary_query_en}",
        f"required_topics: {', '.join(baseline.required_topics)}",
        f"species: {baseline.species}",
        f"emergency_hint: {baseline.emergency_hint}",
    ]
    return "\n".join(lines)

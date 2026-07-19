"""병원 검색 결과 정리 테스트 — 중복·지역·비병원 (명세 33·34절 필수조건 강화).

실제 Tavily 응답에서 관찰된 세 가지 문제를 고정한다.

    1. 같은 병원이 검색어마다 잡혀 목록에 3~4번 나온다.
    2. 서울로 검색했는데 경기(031) 병원이 '추천' 으로 올라온다.
    3. "고양이 건강 체크" 같은 정보성 글이 병원 후보로 남아 점수를 받는다.

세 가지 모두 응급 상황에서 보호자의 시간을 쓰게 만든다. 특히 2번은 한 시간 거리
병원으로 출발하게 만들 수 있어 단순한 품질 문제가 아니다.

**지역 불일치는 표시만 하고 버리지 않는다.** 주소·지역번호 판정이 완전하지 않아
잘못 버리면 그 지역의 유일한 응급 병원이 사라질 수 있다. `region_match` 값을
결과에 남겨 두면 표시 계층이 판단할 수 있다 — 등급 조정은 명세에 없는 판정이라
그래프 안에서 하지 않는다.
"""

from __future__ import annotations

from typing import Any

from petcare_ai.graph.nodes.hospital_search import (
    check_region_match,
    dedupe_hospital_records,
    is_hospital_like,
    normalize_region,
    parse_hospital_results,
)


# ---------------------------------------------------------------------------
# 지역명 파싱 — Android Geocoder 표기와 검색 결과 주소 표기를 같은 축으로 맞춘다
# ---------------------------------------------------------------------------
def test_정식_명칭과_약칭을_같은_시도로_읽는다() -> None:
    assert normalize_region("서울특별시 강남구") == ("서울", "강남구")
    assert normalize_region("서울 강남구") == ("서울", "강남구")
    assert normalize_region("경기도 성남시 분당구") == ("경기", "성남시")
    assert normalize_region("제주특별자치도 제주시") == ("제주", "제주시")


def test_지역명을_못_읽으면_대조를_포기한다() -> None:
    """틀린 기준으로 거르는 것보다 안 거르는 편이 안전하다."""
    assert normalize_region("") == (None, None)
    assert normalize_region(None) == (None, None)
    assert normalize_region("어딘가") == (None, None)


# ---------------------------------------------------------------------------
# 지역 대조
# ---------------------------------------------------------------------------
def test_지역번호가_다른_시도면_불일치다() -> None:
    assert check_region_match({"phone": "031-272-1313"}, "서울") == "mismatch"
    assert check_region_match({"phone": "02-529-5575"}, "서울") == "match"


def test_주소가_있으면_주소가_지역번호를_이긴다() -> None:
    """서울 병원이 경기 대표번호를 쓰는 경우가 있어, 주소를 먼저 본다."""
    record = {"phone": "031-000-0000", "address": "서울 강남구 도산대로 45"}
    assert check_region_match(record, "서울") == "match"


def test_판정할_근거가_없으면_unknown_이다() -> None:
    assert check_region_match({}, "서울") == "unknown"
    assert check_region_match({"phone": "1588-0000"}, "서울") == "unknown"
    # 요청 지역 자체를 모르면 대조하지 않는다.
    assert check_region_match({"phone": "031-272-1313"}, None) == "unknown"


# ---------------------------------------------------------------------------
# 비병원 제외 — '동물병원' 이 한 번 스친 정보성 글을 후보로 세지 않는다
# ---------------------------------------------------------------------------
def test_이름이_병원_형태면_전화번호가_없어도_남긴다() -> None:
    assert is_hospital_like({"name": "미래동물의료센터"}) is True
    assert is_hospital_like({"name": "튼튼동물병원"}) is True


def test_정보성_글은_동물병원을_언급해도_제외한다() -> None:
    blog = {"name": "고양이 건강 체크 방법", "is_animal_hospital": True}
    assert is_hospital_like(blog) is False


def test_병원_이름이_아니어도_연락처가_있으면_남긴다() -> None:
    """지도·디렉터리 페이지는 이름이 깨져 와도 연락처가 있으면 쓸모가 있다."""
    listing = {
        "name": "강남 24시 진료 안내",
        "is_animal_hospital": True,
        "phone": "02-123-4567",
    }
    assert is_hospital_like(listing) is True


# ---------------------------------------------------------------------------
# 중복 병합
# ---------------------------------------------------------------------------
def test_같은_병원의_여러_페이지를_하나로_합치고_빠진_값을_채운다() -> None:
    merged = dedupe_hospital_records(
        [
            {"name": "24시수동물병원", "phone": "02-2676-7582", "source_url": "https://a"},
            {
                "name": "24시 수 동물병원",
                "phone": "02-2676-7582",
                "address": "서울 양천구 목동로 1",
                "source_url": "https://b",
            },
            {"name": "24시수동물병원", "phone": None, "source_url": "https://c"},
        ]
    )

    assert len(merged) == 1
    assert merged[0]["phone"] == "02-2676-7582"
    # 주소는 b 에서 온다 — 지어낸 값이 아니라 이미 검색된 값을 옮긴 것이다.
    assert merged[0]["address"] == "서울 양천구 목동로 1"


def test_분점이_둘_이상이면_연락처_없는_페이지를_아무_지점에나_붙이지_않는다() -> None:
    """어느 지점 소개인지 모르면 합치지 않는다 — 잘못 붙은 주소는 엉뚱한 곳으로 보낸다."""
    merged = dedupe_hospital_records(
        [
            {"name": "24시동물병원", "phone": "02-111-1111", "source_url": "https://a"},
            {"name": "24시동물병원", "phone": "031-222-2222", "source_url": "https://b"},
            {
                "name": "24시동물병원",
                "phone": None,
                "address": "서울 강남구 어딘가로 1",
                "source_url": "https://c",
            },
        ]
    )

    assert len(merged) == 3, "지점을 특정할 수 없으면 흡수하지 않는다."
    # `.get` 을 쓴다 — 병합은 없는 key 를 만들어내지 않는다(값을 지어내지 않는 것과 같은 원칙).
    by_phone = {str(item.get("phone")): item for item in merged}
    assert by_phone["02-111-1111"].get("address") is None
    assert by_phone["031-222-2222"].get("address") is None


def test_이름이_같아도_전화번호가_다르면_분점으로_보고_둘_다_남긴다() -> None:
    """'24시동물병원' 같은 이름은 지역마다 있다. 잘못 합치면 엉뚱한 지점에 전화한다."""
    merged = dedupe_hospital_records(
        [
            {"name": "24시동물병원", "phone": "02-111-1111", "source_url": "https://a"},
            {"name": "24시동물병원", "phone": "031-222-2222", "source_url": "https://b"},
        ]
    )
    assert len(merged) == 2
    assert {item["phone"] for item in merged} == {"02-111-1111", "031-222-2222"}


# ---------------------------------------------------------------------------
# 파싱 전체 경로
# ---------------------------------------------------------------------------
RAW_MIXED: list[dict[str, Any]] = [
    {
        "title": "강남24시동물병원 | 24시간 응급 진료",
        "url": "https://gangnam24vet.example.com",
        "content": "24시간 응급 진료. 전화 02-987-6543. 서울 강남구 언주로 100.",
    },
    {
        # 같은 병원의 블로그 소개 — 중복
        "title": "강남24시동물병원 방문 후기",
        "url": "https://blog.example.com/gangnam24",
        "content": "강남24시동물병원 다녀왔어요. 02-987-6543 으로 전화하면 됩니다.",
    },
    {
        # 정보성 글 — 병원이 아니다
        "title": "고양이 건강 체크 방법 7가지",
        "url": "https://petblog.example.com/cat-health",
        "content": "동물병원에 가기 전 집에서 확인할 수 있는 것들을 정리했습니다.",
    },
    {
        # 지역이 다른 병원 — 남기되 표시한다
        "title": "서울YES동물병원 진료 안내",
        "url": "https://yesvet.example.com",
        "content": "진료 문의 031-272-1313.",
    },
]


def test_파싱이_비병원을_빼고_중복을_합치고_지역을_표시한다() -> None:
    parsed = parse_hospital_results(RAW_MIXED, region_name="서울특별시 강남구")
    by_name = {item["name"]: item for item in parsed}

    assert "고양이 건강 체크 방법 7가지" not in by_name, "정보성 글은 후보가 아니다."
    assert len(parsed) == 2, "같은 병원의 두 페이지는 하나로 합쳐진다."

    assert by_name["강남24시동물병원"]["region_match"] == "match"
    assert by_name["서울YES동물병원"]["region_match"] == "mismatch"


def test_지역을_모르면_지역으로_거르지_않는다() -> None:
    """위치 미확보 상태에서 지역으로 걸러 버리면 후보가 0이 된다."""
    parsed = parse_hospital_results(RAW_MIXED)
    assert len(parsed) == 2
    assert all(item["region_match"] == "unknown" for item in parsed)



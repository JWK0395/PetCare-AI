from __future__ import annotations

from petcare_agent.cli import run_local_harness
from petcare_agent.local_data import (
    DATA_DIR,
    load_local_context,
)


def main() -> None:
    context = load_local_context(DATA_DIR)
    pet_id = int(context["pet"]["id"])

    print("Local JSON 연결 완료")
    print("반려동물:", context["pet"].get("name"))
    print(
        "일기 개수:",
        len(context["daily_entries"]),
    )
    print(
        "진단서 개수:",
        len(context["diagnoses"]),
    )

    run_local_harness(
        context,
        pet_id=pet_id,
    )


if __name__ == "__main__":
    main()

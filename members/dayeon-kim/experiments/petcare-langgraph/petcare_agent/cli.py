from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .local_data import DATA_DIR, load_local_context
from .models import GraphStepResult, PetCareState
from .runtime import resume_petcare, start_petcare


def append_transcript(
    path: Path,
    record: dict[str, Any],
) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


def print_step_metadata(
    step: GraphStepResult,
    *,
    trace: list[str],
) -> None:
    state = step["state"]

    if step["status"] == "waiting_for_user":
        route = "question_manager"
        risk = "unknown"
        needs_user_response = True
    else:
        route = state.get("route")
        risk = state.get("route")
        needs_user_response = False

    triage_status = state.get(
        "triage_status",
        "idle",
    )

    print(
        f"[route={route} "
        f"risk={risk} "
        f"triage={triage_status} "
        f"needs_user_response={needs_user_response}]"
    )
    print("[trace=" + " -> ".join(trace) + "]")


def run_local_harness(
    backend_context: dict[str, Any],
    *,
    pet_id: int = 103,
) -> None:
    conversation_id = (
        f"local_{datetime.now():%Y%m%d_%H%M%S}"
    )

    transcript_dir = Path("tmp")
    transcript_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transcript_path = (
        transcript_dir
        / f"agent-harness_{conversation_id}.jsonl"
    )

    pending_interrupt = False
    accumulated_trace: list[str] = []
    last_state: PetCareState = {}

    print("Agent: petcare-assessment-graph")
    print("Data source: backend-context-snapshot")
    print(f"Pet id: {pet_id}")
    print(f"Conversation id: {conversation_id}")
    print(f"Transcript: {transcript_path}")
    print("Type a message, or use /help for commands.")

    while True:
        try:
            user_text = input("\nUser> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not user_text:
            continue

        command = user_text.lower()

        if command in {
            "/quit",
            "/exit",
            "quit",
            "exit",
        }:
            print("종료합니다.")
            break

        if command == "/help":
            print(
                "/help: 도움말 | "
                "/state: 현재 State | "
                "/memory: 최근 대화 | "
                "/quit: 종료"
            )
            continue

        if command == "/memory":
            print(
                json.dumps(
                    last_state.get(
                        "conversation_history",
                        [],
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            continue

        if command == "/reload":
            try:
                backend_context = load_local_context(
                    DATA_DIR
                )
                pet_id = int(
                    backend_context["pet"]["id"]
                )
                print(
                    "JSON을 다시 읽었습니다:",
                    backend_context["pet"].get("name"),
                    f"(pet_id={pet_id})",
                )
            except Exception as error:
                print(
                    "JSON 다시 읽기 실패:",
                    type(error).__name__,
                    error,
                )
            continue

        if command == "/state":
            state_preview = {
                "route": last_state.get("route"),
                "triage_status": last_state.get(
                    "triage_status",
                    "idle",
                ),
                "post_triage_mode": last_state.get(
                    "post_triage_mode",
                    False,
                ),
                "conversation_history": last_state.get(
                    "conversation_history",
                    [],
                ),
                "question_strategy": last_state.get(
                    "question_strategy",
                    {},
                ),
                "follow_up_history": last_state.get(
                    "follow_up_history",
                    [],
                ),
                "emergency_hits": last_state.get(
                    "emergency_hits",
                    [],
                ),
                "recovery_hits": last_state.get(
                    "recovery_hits",
                    [],
                ),
                "rag_done": last_state.get(
                    "rag_done",
                    False,
                ),
                "errors": last_state.get("errors", []),
            }

            print(
                json.dumps(
                    state_preview,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            continue

        append_transcript(
            transcript_path,
            {
                "timestamp": datetime.now().isoformat(),
                "role": "user",
                "content": user_text,
                "is_resume": pending_interrupt,
            },
        )

        if pending_interrupt:
            step = resume_petcare(
                session_id=conversation_id,
                answer=user_text,
            )
        else:
            accumulated_trace = []

            request_payload = {
                "session_id": conversation_id,
                "pet_id": pet_id,
                "user_input": user_text,
                "context": backend_context,
            }

            step = start_petcare(request_payload)

        last_state = step["state"]

        for trace_item in step["trace"]:
            if (
                not accumulated_trace
                or accumulated_trace[-1] != trace_item
            ):
                accumulated_trace.append(trace_item)

        if step["status"] == "waiting_for_user":
            assistant_text = (
                step["question"]
                or "추가 확인이 필요합니다."
            )
            pending_interrupt = True
            response_type = "follow_up"
        else:
            assistant_text = (
                step["state"].get("answer")
                or "답변을 생성하지 못했습니다."
            )
            pending_interrupt = False
            response_type = "final"

        print(f"Assistant> {assistant_text}")
        print_step_metadata(
            step,
            trace=accumulated_trace,
        )

        append_transcript(
            transcript_path,
            {
                "timestamp": datetime.now().isoformat(),
                "role": "assistant",
                "type": response_type,
                "content": assistant_text,
                "status": step["status"],
                "route": step["state"].get("route"),
                "question_field": step.get("field"),
                "trace": accumulated_trace,
                "conversation_history": (
                    step["state"].get(
                        "conversation_history",
                        [],
                    )
                ),
                "latency_ms": step["state"].get(
                    "latency_ms",
                    {},
                ),
                "errors": step["state"].get(
                    "errors",
                    [],
                ),
            },
        )

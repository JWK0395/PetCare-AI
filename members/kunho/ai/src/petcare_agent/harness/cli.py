"""Console entrypoint for testing PetCare agent implementations locally."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from petcare_agent.harness.adapter import AgentSessionConfig, AgentTurnResult
from petcare_agent.harness.data_bundle import DataBundle
from petcare_agent.harness.fake_backend import (
    DataBundleBackendProvider,
    DataBundleRAGAdapter,
)
from petcare_agent.harness.registry import available_agent_names, load_agent_adapter


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    bundle = DataBundle.load(args.data_zip)
    if args.list_pets:
        _print_pet_list(bundle)
        return 0

    conversation_id = args.conversation_id or _default_conversation_id(args.timezone)
    transcript_path = None
    if not args.no_transcript:
        transcript_path = Path(args.transcript_dir) / f"{conversation_id}.jsonl"

    adapter = load_agent_adapter(args.agent)
    session = adapter.start_session(
        config=AgentSessionConfig(
            pet_id=args.pet_id,
            conversation_id=conversation_id,
            locale=args.locale,
            timezone=args.timezone,
            db_context_days=args.days,
            rag_top_k=args.rag_top_k,
        ),
        context_provider=DataBundleBackendProvider(bundle),
        rag_adapter=DataBundleRAGAdapter(bundle),
    )

    _print_banner(args, adapter.name, conversation_id, transcript_path)

    if args.once:
        result = _run_turn(session, args.once, transcript_path)
        _print_turn(result)
        return 0

    if args.replay:
        for message in _iter_replay_messages(Path(args.replay)):
            print(f"\nUser> {message}")
            result = _run_turn(session, message, transcript_path)
            _print_turn(result)
        return 0

    return _interactive_loop(session, transcript_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PetCare agent against a local data.zip or unpacked fixture "
            "bundle, then chat in the console."
        )
    )
    parser.add_argument(
        "--data-zip",
        required=True,
        help="Path to data.zip or an unpacked data bundle directory.",
    )
    parser.add_argument("--pet-id", type=int, required=True, help="Pet id to test.")
    parser.add_argument(
        "--agent",
        default="current-assessment-graph",
        help=(
            "Agent adapter name or module:attribute. Built-ins: "
            + ", ".join(available_agent_names())
        ),
    )
    parser.add_argument("--days", type=int, default=3, help="Recent context window.")
    parser.add_argument("--rag-top-k", type=int, default=5, help="RAG chunks to retrieve.")
    parser.add_argument("--locale", default="ko-KR")
    parser.add_argument("--timezone", default="Asia/Seoul")
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument(
        "--transcript-dir",
        default=".tmp/agent-harness",
        help="Directory for JSONL transcripts.",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Do not write a JSONL transcript.",
    )
    parser.add_argument(
        "--once",
        help="Run one message and exit. Useful for smoke tests.",
    )
    parser.add_argument(
        "--replay",
        help="Replay a text or JSONL transcript file and exit.",
    )
    parser.add_argument(
        "--list-pets",
        action="store_true",
        help="List pet records found in the bundle and exit.",
    )
    return parser


def _interactive_loop(session: Any, transcript_path: Path | None) -> int:
    print("Type a message, or use /help for local harness commands.")
    while True:
        try:
            user_input = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        command = _handle_command(session, user_input)
        if command == "exit":
            return 0
        if command == "handled":
            continue

        try:
            result = _run_turn(session, user_input, transcript_path)
        except Exception as exc:
            print(f"Harness error: {exc}")
            continue
        _print_turn(result)


def _handle_command(session: Any, user_input: str) -> str | None:
    command = user_input.strip()
    if command in {"/exit", "/quit"}:
        return "exit"
    if command == "/help":
        print(
            "Commands: /help, /state, /handoff, /visit yes|no|undecided|not_asked, "
            "/exit"
        )
        return "handled"
    if command == "/state":
        state = getattr(session, "state", None)
        if state is None:
            print("This adapter does not expose state.")
        else:
            _print_json(_state_summary(state))
        return "handled"
    if command == "/handoff":
        state = getattr(session, "state", None)
        if state is None:
            print("This adapter does not expose handoff state.")
        else:
            _print_json(state.handoff.model_dump(mode="json"))
        return "handled"
    if command.startswith("/visit "):
        intent = command.removeprefix("/visit ").strip()
        setter = getattr(session, "set_hospital_visit_intent", None)
        if setter is None:
            print("This adapter does not support local visit intent commands.")
        else:
            try:
                setter(intent)
                print(f"hospital_visit_intent set to {intent}")
            except ValueError as exc:
                print(str(exc))
        return "handled"
    return None


def _run_turn(
    session: Any,
    message: str,
    transcript_path: Path | None,
) -> AgentTurnResult:
    result = session.handle_user_message(message)
    if transcript_path is not None:
        _append_transcript(transcript_path, message, result)
    return result


def _print_turn(result: AgentTurnResult) -> None:
    response = result.response
    print(f"Assistant> {response.assistant_message}")
    print(
        f"[route={response.route} risk={response.risk_level} "
        f"needs_user_response={response.needs_user_response}]"
    )
    if result.trace_path:
        print(f"[trace={' -> '.join(result.trace_path)}]")
    if response.follow_up_question is not None:
        print(f"[follow_up_question={response.follow_up_question.question_id}]")
    if response.handoff.type != "none":
        print(f"[handoff={response.handoff.type}]")
        if response.handoff.summary:
            print(response.handoff.summary)
        if response.handoff.email_draft:
            print(response.handoff.email_draft)


def _append_transcript(
    transcript_path: Path,
    user_input: str,
    result: AgentTurnResult,
) -> None:
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "user_input": user_input,
        "response": result.response.model_dump(mode="json"),
        "trace_path": result.trace_path,
        "state_summary": _state_summary(result.state),
        "fallback_reason": result.fallback_reason,
    }
    with transcript_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _state_summary(state: Any) -> dict[str, Any]:
    return {
        "intent": state.intent,
        "species": state.species,
        "risk_level": state.risk_level,
        "confidence": state.confidence,
        "next_route": state.next_route,
        "safety_question_turns": state.safety_question_turns,
        "hospital_visit_intent": state.hospital_visit_intent,
        "context": {
            "pet_id": state.context.pet.get("id") or state.context.pet.get("pet_id"),
            "daily_entries": len(state.context.recent_daily_entries),
            "diagnoses": len(state.context.diagnoses),
            "unknown_items": list(state.context.unknown_items),
            "data_from": state.context.data_from,
            "data_to": state.context.data_to,
        },
        "assessment": state.assessment.model_dump(mode="json"),
        "change_detection": state.change_detection.model_dump(mode="json"),
        "triggered_rules": [
            rule.rule_id for rule in state.emergency_screening.triggered_rules
        ],
    }


def _iter_replay_messages(path: Path) -> list[str]:
    messages: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            payload = json.loads(line)
            message = payload.get("user_input") or payload.get("message")
            if message:
                messages.append(str(message))
            continue
        messages.append(line)
    return messages


def _print_pet_list(bundle: DataBundle) -> None:
    if not bundle.pets:
        print("No db/pets.json records were found.")
        return
    for pet in bundle.pets:
        pet_id = pet.get("id") or pet.get("pet_id")
        name = pet.get("name") or "(unnamed)"
        species = pet.get("species") or "unknown"
        print(f"{pet_id}\t{name}\t{species}")


def _print_banner(
    args: argparse.Namespace,
    agent_name: str,
    conversation_id: str,
    transcript_path: Path | None,
) -> None:
    print(f"Agent: {agent_name}")
    print(f"Data bundle: {args.data_zip}")
    print(f"Pet id: {args.pet_id}")
    print(f"Conversation id: {conversation_id}")
    if transcript_path is not None:
        print(f"Transcript: {transcript_path}")


def _default_conversation_id(timezone_name: str) -> str:
    try:
        now = datetime.now(ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError:
        now = datetime.now()
    return f"local_{now.strftime('%Y%m%d_%H%M%S')}"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

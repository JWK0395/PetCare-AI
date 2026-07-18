from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
import os
import smtplib
import uuid
from typing import Any, Callable, Protocol

from openai import OpenAI
from pydantic import BaseModel

from .config import Settings
from .models import (
    EmailDeliveryResult,
    HospitalInfo,
    RAGChunk,
)


class OpenAIService:
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key
        )
        self.model = model

    def parse(
        self,
        *,
        schema: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
    ) -> BaseModel:
        response = (
            self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                text_format=schema,
            )
        )

        if response.output_parsed is None:
            raise RuntimeError(
                "OpenAI 구조화 출력이 "
                "비어 있습니다."
            )

        return response.output_parsed

    def text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = (
            self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )
        )

        if not response.output_text:
            raise RuntimeError(
                "OpenAI 텍스트 출력이 "
                "비어 있습니다."
            )

        return response.output_text.strip()


_llm_service: OpenAIService | Any | None = (
    None
)


def get_llm_service() -> OpenAIService:
    global _llm_service

    if _llm_service is None:
        settings = Settings.from_env()
        _llm_service = OpenAIService(
            api_key=(
                settings.openai_api_key
            ),
            model=settings.openai_model,
        )

    return _llm_service


def set_llm_service(
    service: OpenAIService | Any,
) -> None:
    global _llm_service
    _llm_service = service


class RAGProvider(Protocol):
    def search(
        self,
        *,
        query: str,
        pet_context: dict[str, Any],
        limit: int = 5,
    ) -> list[RAGChunk]:
        ...


class DemoRAGProvider:
    def __init__(self) -> None:
        self.documents = [
            RAGChunk(
                source_id="guide-001",
                title=(
                    "검수 완료 반려동물 "
                    "구토 보호자 안내"
                ),
                organization=(
                    "PetCare AI Demo"
                ),
                version="2026.1",
                page=3,
                text=(
                    "구토가 반복되거나 "
                    "기력 저하, 식욕 감소와 "
                    "함께 나타나는 경우에는 "
                    "동물병원 상담이 권장된다."
                ),
                score=0.95,
                metadata={
                    "species": [
                        "dog",
                        "cat",
                    ],
                    "topic": "vomiting",
                },
            )
        ]

    def search(
        self,
        *,
        query: str,
        pet_context: dict[str, Any],
        limit: int = 5,
    ) -> list[RAGChunk]:
        return self.documents[:limit]


class TeamRAGAdapter:
    def __init__(
        self,
        search_function: Callable[
            ...,
            list[dict[str, Any]],
        ],
    ) -> None:
        self.search_function = (
            search_function
        )

    def search(
        self,
        *,
        query: str,
        pet_context: dict[str, Any],
        limit: int = 5,
    ) -> list[RAGChunk]:
        results = self.search_function(
            query=query,
            pet_context=pet_context,
            limit=limit,
        )
        return [
            RAGChunk.model_validate(
                item
            )
            for item in results
        ]


_rag_provider: RAGProvider = (
    DemoRAGProvider()
)


def get_rag_provider() -> RAGProvider:
    return _rag_provider


def set_rag_provider(
    provider: RAGProvider,
) -> None:
    global _rag_provider
    _rag_provider = provider


class HospitalSearchProvider(Protocol):
    def search_open(
        self,
        *,
        location: dict[str, Any] | None,
        limit: int = 5,
    ) -> list[HospitalInfo]:
        ...


class DemoHospitalSearchProvider:
    def search_open(
        self,
        *,
        location: dict[str, Any] | None,
        limit: int = 5,
    ) -> list[HospitalInfo]:
        address = (
            location.get("address")
            if location
            else None
        )

        return [
            HospitalInfo(
                hospital_id="demo-001",
                name="가까운 24시 동물병원",
                address=(
                    address
                    or "현재 위치 인근"
                ),
                phone="051-000-0000",
                email=(
                    "emergency@example.com"
                ),
                distance_km=1.2,
                is_open=True,
                open_status="운영 중",
                source="demo",
            )
        ][:limit]


class TeamHospitalSearchAdapter:
    def __init__(
        self,
        search_function: Callable[
            ...,
            list[dict[str, Any]],
        ],
    ) -> None:
        self.search_function = (
            search_function
        )

    def search_open(
        self,
        *,
        location: dict[str, Any] | None,
        limit: int = 5,
    ) -> list[HospitalInfo]:
        results = self.search_function(
            location=location,
            open_now=True,
            limit=limit,
        )

        return [
            HospitalInfo.model_validate(
                item
            )
            for item in results
        ]


_hospital_provider: (
    HospitalSearchProvider
) = DemoHospitalSearchProvider()


def get_hospital_search_provider(
) -> HospitalSearchProvider:
    return _hospital_provider


def set_hospital_search_provider(
    provider: HospitalSearchProvider,
) -> None:
    global _hospital_provider
    _hospital_provider = provider


class EmailProvider(Protocol):
    def send(
        self,
        *,
        recipient: str,
        subject: str,
        body: str,
    ) -> EmailDeliveryResult:
        ...



class OutboxEmailProvider:
    def __init__(
        self,
        output_dir: str | Path = (
            "tmp/outbox"
        ),
    ) -> None:
        self.output_dir = Path(
            output_dir
        )

    def send(
        self,
        *,
        recipient: str,
        subject: str,
        body: str,
    ) -> EmailDeliveryResult:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        message_id = uuid.uuid4().hex
        eml_path = (
            self.output_dir
            / f"{message_id}.eml"
        )
        preview_path = (
            self.output_dir
            / f"{message_id}_preview.txt"
        )

        message = EmailMessage()
        message["From"] = (
            "petcare-local@example.com"
        )
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(
            body,
            charset="utf-8",
        )

        eml_path.write_bytes(
            message.as_bytes()
        )

        preview_path.write_text(
            "\n".join(
                [
                    f"From: {message['From']}",
                    f"To: {recipient}",
                    f"Subject: {subject}",
                    "",
                    body,
                ]
            ),
            encoding="utf-8",
        )

        return EmailDeliveryResult(
            status="saved",
            recipient=recipient,
            message_id=message_id,
            file_path=str(eml_path),
            preview_path=str(
                preview_path
            ),
        )


class SMTPEmailProvider:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        sender: str,
        use_tls: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.sender = sender
        self.use_tls = use_tls

    @classmethod
    def from_env(
        cls,
    ) -> "SMTPEmailProvider":
        required = {
            "SMTP_HOST": os.getenv(
                "SMTP_HOST"
            ),
            "SMTP_USERNAME": os.getenv(
                "SMTP_USERNAME"
            ),
            "SMTP_PASSWORD": os.getenv(
                "SMTP_PASSWORD"
            ),
            "SMTP_SENDER": os.getenv(
                "SMTP_SENDER"
            ),
        }

        missing = [
            name
            for name, value
            in required.items()
            if not value
        ]

        if missing:
            raise ValueError(
                "SMTP 환경변수가 "
                f"필요합니다: {missing}"
            )

        return cls(
            host=str(
                required["SMTP_HOST"]
            ),
            port=int(
                os.getenv(
                    "SMTP_PORT",
                    "587",
                )
            ),
            username=str(
                required[
                    "SMTP_USERNAME"
                ]
            ),
            password=str(
                required[
                    "SMTP_PASSWORD"
                ]
            ),
            sender=str(
                required["SMTP_SENDER"]
            ),
            use_tls=(
                os.getenv(
                    "SMTP_USE_TLS",
                    "true",
                ).lower()
                != "false"
            ),
        )

    def send(
        self,
        *,
        recipient: str,
        subject: str,
        body: str,
    ) -> EmailDeliveryResult:
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        with smtplib.SMTP(
            self.host,
            self.port,
            timeout=20,
        ) as smtp:
            if self.use_tls:
                smtp.starttls()

            smtp.login(
                self.username,
                self.password,
            )
            smtp.send_message(message)

        return EmailDeliveryResult(
            status="sent",
            recipient=recipient,
            message_id=(
                message.get(
                    "Message-ID"
                )
            ),
        )


_email_provider: EmailProvider | None = (
    None
)


def get_email_provider() -> EmailProvider:
    global _email_provider

    if _email_provider is not None:
        return _email_provider

    if os.getenv("SMTP_HOST"):
        _email_provider = (
            SMTPEmailProvider.from_env()
        )
    else:
        _email_provider = (
            OutboxEmailProvider()
        )

    return _email_provider


def set_email_provider(
    provider: EmailProvider,
) -> None:
    global _email_provider
    _email_provider = provider

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request
from urllib.parse import urljoin


class EmbeddingsAPI(Protocol):
    def create(self, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class EmbeddingResponse:
    embeddings: list[list[float]]


def create_embeddings(
    embeddings_api: EmbeddingsAPI,
    model: str,
    inputs: list[str],
) -> EmbeddingResponse:
    if not inputs:
        return EmbeddingResponse([])

    response = embeddings_api.create(model=model, input=inputs)
    return EmbeddingResponse(_response_embeddings(response))


class HashEmbeddingAPI:
    """Small deterministic embedding provider for tests and offline demos."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        self._dimensions = dimensions

    def create(self, **kwargs: object) -> object:
        raw_input = kwargs.get("input", [])
        if isinstance(raw_input, str):
            inputs = [raw_input]
        elif isinstance(raw_input, list):
            inputs = [str(item) for item in raw_input]
        else:
            inputs = []

        return {
            "data": [
                {
                    "embedding": _hash_embedding(text, self._dimensions),
                    "index": index,
                }
                for index, text in enumerate(inputs)
            ]
        }


class OllamaEmbeddingAPI:
    """Embedding provider backed by a local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = _normalize_base_url(base_url)

    def create(self, **kwargs: object) -> object:
        model = kwargs.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Ollama embedding model must be a non-empty string.")

        raw_input = kwargs.get("input", [])
        if isinstance(raw_input, str):
            inputs = [raw_input]
        elif isinstance(raw_input, list):
            inputs = [str(item) for item in raw_input]
        else:
            inputs = []

        payload = json.dumps(
            {
                "model": model,
                "input": inputs,
            }
        ).encode("utf-8")
        http_request = request.Request(
            urljoin(self._base_url, "/api/embed"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Ollama embeddings request failed. "
                "Make sure Ollama is running and the embedding model is pulled."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned an invalid JSON response.") from exc

        embeddings = response_payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama response did not include embeddings.")

        return {
            "data": [
                {
                    "embedding": embedding,
                    "index": index,
                }
                for index, embedding in enumerate(embeddings)
            ]
        }


def _response_embeddings(response: object) -> list[list[float]]:
    data = response.get("data") if isinstance(response, dict) else getattr(response, "data")
    embeddings: list[list[float]] = []
    for item in data:
        raw_embedding = (
            item.get("embedding")
            if isinstance(item, dict)
            else getattr(item, "embedding")
        )
        embeddings.append([float(value) for value in raw_embedding])
    return embeddings


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return vector
    return [value / length for value in vector]


def _normalize_base_url(base_url: str) -> str:
    stripped = base_url.strip().rstrip("/")
    if not stripped:
        return "http://localhost:11434"
    if "://" not in stripped:
        return f"http://{stripped}"
    return stripped


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in "".join(
            character if character.isalnum() else " "
            for character in text
        ).split()
    ]

import logging
import os
import json
import pathlib

from flax import serialization
import flax.nnx as nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download
import openpi.shared.normalize as _normalize

PALIGEMMA_VOCAB_SIZE = 257_152
FORCE_VQ_TOKEN_OFFSET_BASE = 128
SKILL_TOKEN_OFFSET_BASE = 1024




class PaligemmaTokenizer:
    def __init__(
        self,
        max_len: int = 48,
    ):
        self._max_len = max_len
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def encode_text(self, text: str, *, add_bos: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_bos=add_bos)

    def decode_token_ids(self, token_ids: np.ndarray | list[int]) -> str:
        token_ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        token_ids = token_ids[(token_ids > 0) & (token_ids < self._tokenizer.get_piece_size())]
        if token_ids.size == 0:
            return ""
        return self._tokenizer.decode(token_ids.tolist())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


    def _finalize_sequence(
        self,
        tokens: list[int],
        *,
        ar_mask: list[int] | None = None,
        loss_mask: list[bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        token_mask = [True] * len(tokens)
        if ar_mask is not None and len(ar_mask) != len(tokens):
            raise ValueError(f"`ar_mask` length ({len(ar_mask)}) must match token length ({len(tokens)}).")
        if loss_mask is not None and len(loss_mask) != len(tokens):
            raise ValueError(f"`loss_mask` length ({len(loss_mask)}) must match token length ({len(tokens)}).")

        if len(tokens) < self._max_len:
            pad_len = self._max_len - len(tokens)
            tokens = tokens + [0] * pad_len
            token_mask = token_mask + [False] * pad_len
            if ar_mask is not None:
                ar_mask = ar_mask + [0] * pad_len
            if loss_mask is not None:
                loss_mask = loss_mask + [False] * pad_len
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            if ar_mask is not None:
                ar_mask = ar_mask[: self._max_len]
            if loss_mask is not None:
                loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens, dtype=np.int32),
            np.asarray(token_mask, dtype=bool),
            None if ar_mask is None else np.asarray(ar_mask, dtype=np.int32),
            None if loss_mask is None else np.asarray(loss_mask, dtype=bool),
        )



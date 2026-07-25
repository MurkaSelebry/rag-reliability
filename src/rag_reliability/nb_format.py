"""Self-contained SFT record builder mirrored verbatim into the training notebook.

The notebook imports this when it can clone the repo; when it can't, it pastes
INLINE_SOURCE. tests/test_nb_format.py asserts this stays equal to the repo's
build_chat_training_record so the notebook fallback never drifts.
"""

from __future__ import annotations

from rag_reliability.formatting import build_chat_training_record
from rag_reliability.schema import RagSample


def build_sft_messages(sample: RagSample, mode: str) -> dict[str, list[dict[str, str]]]:
    """One SFT chat record: user=judge prompt, assistant=JSON verdict."""
    return build_chat_training_record(sample, mode)

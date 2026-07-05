"""Structured output schemas for the agent."""

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """A single docs-grounded answer.

    Citations and referrals join this schema in M2, once retrieval exists.
    """

    answer_md: str = Field(description="The answer in markdown; may embed code snippets.")
    symbols_used: list[str] = Field(
        default_factory=list,
        description="Every PyTorch API symbol the answer relies on, e.g. 'torch.optim.SGD'.",
    )
    torch_version: str = Field(
        default="unknown",
        description="The PyTorch version the answer targets, e.g. '2.12'.",
    )

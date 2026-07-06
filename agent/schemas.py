"""Structured output schemas for the agent."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """The answer came from here — a live docs URL, copied from the context."""

    url: str = Field(description="Exact URL of a provided context section.")
    anchor: str = Field(default="", description="Exact anchor of that section.")
    title: str = Field(default="", description="Heading path of that section.")


class Referral(BaseModel):
    """This is beyond the docs — look here (source link, search, DeepWiki)."""

    url: str
    reason: str = Field(default="", description="Why the user should look there.")


class Answer(BaseModel):
    """A single docs-grounded answer."""

    answer_md: str = Field(description="The answer in markdown; may embed code snippets.")
    symbols_used: list[str] = Field(
        default_factory=list,
        description="Every PyTorch API symbol the answer relies on, e.g. 'torch.optim.SGD'.",
    )
    torch_version: str = Field(
        default="unknown",
        description="The PyTorch version the answer targets, e.g. '2.12'.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Context sections the answer is based on — url/anchor copied exactly.",
    )
    referrals: list[Referral] = Field(
        default_factory=list,
        description="Where to look for what the context does not cover.",
    )

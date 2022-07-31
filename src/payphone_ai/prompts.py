import dataclasses
import functools
import random
import types
from typing import Collection, List, Literal, Mapping, Tuple, TypedDict

import boto3
import jmespath


@dataclasses.dataclass(frozen=True)
class Prompt:
    text: str
    constraints: Tuple[str, ...] = dataclasses.field(
        default=("LanguageCode == 'en-US'",),
        kw_only=True,
    )
    speed: Literal["x-slow", "slow", "medium", "fast", "x-fast"] = dataclasses.field(
        default="medium",
        kw_only=True,
    )


@dataclasses.dataclass(frozen=True)
class ResolvedPrompt(Prompt):
    voice_id: str = dataclasses.field(kw_only=True)
    engine: str = dataclasses.field(kw_only=True)


class Voice(TypedDict):
    Id: str
    SupportedEngines: List[str]


PROMPTS = (
    Prompt(
        "The caller is Santa Claus, and needs to warn the recipient that they've been very naughty.",
        constraints=(
            "LanguageCode == 'en-US'",
            "Gender == 'Male'",
        ),
    ),
    Prompt(
        "The caller wants to tell the famous joke 'The Aristocrats'.",
    ),
    Prompt(
        "The caller wants to know if the recipient is their mommy, and is insistent that they must be their mommy.",
        constraints=("contains(['Ivy', 'Justin', 'Kevin'], Id)",),
    ),
    Prompt(
        "The caller needs directions to the nearest Pizza Hut.",
    ),
    Prompt(
        "The caller needs help with their math homework.",
        constraints=("contains(['Ivy', 'Justin', 'Kevin'], Id)",),
    ),
    Prompt(
        "The caller has misplaced an item and needs help finding it.",
    ),
    Prompt(
        "The caller warns the recipient not to look now, but they're being followed.",
        speed="fast",
    ),
    Prompt(
        "The caller is conducting a phone interview for a content marketing role.",
    ),
)


@functools.cache
def describe_voices() -> Collection[Voice]:
    client = boto3.client("polly")
    voices: List[Voice] = []
    for voice in client.get_paginator("describe_voices").paginate().search("Voices"):
        voices.append(voice)
    return tuple(voices)


def determine_allowed_voices_for_prompt(prompt: Prompt) -> Collection[Voice]:
    voices = []
    for voice in describe_voices():
        passes_constraints = all(
            jmespath.search(constraint, voice) for constraint in prompt.constraints
        )
        if passes_constraints:
            voices.append(voice)
    if not voices:
        raise ValueError(f"Prompt is overconstrained: {prompt}")
    return tuple(voices)


@functools.cache
def determine_allowed_voices() -> Mapping[Prompt, Collection[Voice]]:
    result = {}
    for prompt in PROMPTS:
        result[prompt] = determine_allowed_voices_for_prompt(prompt)
    return types.MappingProxyType(result)


def get_random_prompt() -> ResolvedPrompt:
    prompt, voices = random.choice(list(determine_allowed_voices().items()))
    voice = random.choice(list(voices))
    return ResolvedPrompt(
        **dataclasses.asdict(prompt),
        voice_id=voice["Id"],
        engine=random.choice(voice["SupportedEngines"]),
    )

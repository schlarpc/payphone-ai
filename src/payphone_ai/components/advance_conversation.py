import contextlib
import logging
import os
import random

import nltk
import openai

logger = logging.getLogger(__name__)


SCENARIOS = [
    "The caller is Santa Claus, and needs to warn the recipient that they've been very naughty.",
    "The caller wants to tell the famous joke 'The Aristocrats'.",
    "The caller wants to know if the recipient is their mommy, and is insistent that they must be their mommy.",
    "The caller needs directions to the nearest Pizza Hut.",
    "The caller needs help with their math homework.",
    "The caller has misplaced an item and needs help finding it.",
    "The caller warns the recipient not to look now, but they're being followed.",
    "The caller is conducting a phone interview for a content marketing role.",
]


def get_random_scenario():
    return random.choice(SCENARIOS)


@contextlib.contextmanager
def start_ai_conversation():
    prompt = f"""\
    The following is a phone conversation.
    {get_random_scenario()}
    """
    users = ["Caller", "Receiver"]
    start_sequence = f"\n{users[0]}: "
    restart_sequence = f"\n{users[1]}: "
    try:

        def _run_ai_conversation_turn(input_text):
            nonlocal prompt
            prompt += restart_sequence + input_text + start_sequence
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                temperature=0.9,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0.9,
                presence_penalty=0.6,
                stop=users,
                stream=True,
                api_key=os.environ["OPENAI_API_KEY"],
            )
            for chunk in response:
                output_text_chunk = chunk["choices"][0]["text"]
                prompt += output_text_chunk
                yield output_text_chunk

        def _iterate_ai_conversation_turn(input_text):
            buffer = ""
            for output_text in _run_ai_conversation_turn(input_text):
                buffer += output_text
                sentences = nltk.tokenize.sent_tokenize(buffer)
                if not sentences:
                    continue
                buffer = sentences.pop()
                for sentence in sentences:
                    yield sentence.strip()
            yield buffer.strip()

        yield _iterate_ai_conversation_turn
    finally:
        pass


async def main(human_text_receiver, ai_text_sender):
    async with human_text_receiver, ai_text_sender:
        with start_ai_conversation() as ai_conversation:
            async for human_text in human_text_receiver:
                logger.info("Human: %s", human_text)
                for ai_text in ai_conversation(human_text):
                    logger.info("AI: %s", ai_text)
                    await ai_text_sender.send(ai_text)

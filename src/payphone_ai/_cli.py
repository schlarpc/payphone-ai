import json
import logging
import os
import uuid
from typing import Tuple, TypedDict, cast
from xml.etree.ElementTree import QName

import httpx
import trio
import trio_asyncio
import trio_websocket
from isort import stream

from . import components, prompts

logger = logging.getLogger()


class StreamMetadata(TypedDict):
    streamSid: str
    accountSid: str
    callSid: str


class SimpleChannel:
    def __init__(self):
        self._channels: Tuple[
            trio.MemorySendChannel, trio.MemoryReceiveChannel
        ] = trio.open_memory_channel(max_buffer_size=0)

    @property
    def sender(self):
        return self._channels[0]

    @property
    def receiver(self):
        return self._channels[1]


async def get_stream_metadata(websocket) -> StreamMetadata:
    while True:
        message = await websocket.get_message()
        data = json.loads(message)
        if data["event"] == "start":
            return cast(StreamMetadata, data["start"])
        elif data["event"] not in {"connected"}:
            raise Exception(f"Got unexpected event at startup: {data['event']}")


async def end_call(stream_metadata: StreamMetadata):
    logger.info("Ending call %s", stream_metadata["callSid"])
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.twilio.com/"
            + "/".join(
                (
                    "2010-04-01",
                    "Accounts",
                    stream_metadata["accountSid"],
                    "Calls",
                    f"{stream_metadata['callSid']}.json",
                )
            ),
            data={"Status": "completed"},
            auth=(os.environ["TWILIO_API_SID"], os.environ["TWILIO_API_SECRET"]),
        )


async def handler(request: trio_websocket.WebSocketRequest) -> None:
    session_id = uuid.uuid4()
    logger.info("Session started: %s", session_id)
    prompt = prompts.get_random_prompt()
    try:
        websocket: trio_websocket.WebSocketConnection = await request.accept()
        stream_metadata = await get_stream_metadata(websocket)
        try:
            async with trio.open_nursery() as nursery:
                human_audio = SimpleChannel()
                human_text = SimpleChannel()
                ai_text = SimpleChannel()
                ai_audio = SimpleChannel()

                nursery.start_soon(
                    components.handle_twilio_events.main,
                    websocket,
                    human_audio.sender,
                )

                nursery.start_soon(
                    components.transcribe_human_audio.main,
                    human_audio.receiver,
                    human_text.sender,
                )

                nursery.start_soon(
                    components.advance_conversation.main,
                    prompt,
                    human_text.receiver,
                    ai_text.sender,
                )

                nursery.start_soon(
                    components.synthesize_ai_audio.main,
                    prompt,
                    ai_text.receiver,
                    ai_audio.sender,
                )

                nursery.start_soon(
                    components.send_ai_audio.main,
                    websocket,
                    stream_metadata,
                    ai_audio.receiver,
                )
        finally:
            await end_call(stream_metadata)
    except Exception:
        pass
    finally:
        logger.info("Session ended: %s", session_id)


async def trio_main():
    logging.basicConfig(level=logging.INFO)
    await trio_websocket.serve_websocket(handler, host=None, port=5000, ssl_context=None)


def main():
    trio_asyncio.run(trio_main)

import logging
import uuid
from typing import Tuple

import trio
import trio_asyncio
import trio_websocket

from . import components

logger = logging.getLogger()


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


async def handler(request: trio_websocket.WebSocketRequest) -> None:
    session_id = uuid.uuid4()
    logger.info("Session started: %s", session_id)
    try:
        websocket: trio_websocket.WebSocketConnection = await request.accept()
        async with trio.open_nursery() as nursery:
            stream_sid = SimpleChannel()
            human_audio = SimpleChannel()
            human_text = SimpleChannel()
            ai_text = SimpleChannel()
            ai_audio = SimpleChannel()

            nursery.start_soon(
                components.handle_twilio_events.main,
                websocket,
                stream_sid.sender,
                human_audio.sender,
            )

            nursery.start_soon(
                components.transcribe_human_audio.main,
                human_audio.receiver,
                human_text.sender,
            )

            nursery.start_soon(
                components.advance_conversation.main,
                human_text.receiver,
                ai_text.sender,
            )

            nursery.start_soon(
                components.synthesize_ai_audio.main,
                ai_text.receiver,
                ai_audio.sender,
            )

            nursery.start_soon(
                components.send_ai_audio.main,
                websocket,
                stream_sid.receiver,
                ai_audio.receiver,
            )
    except Exception:
        pass
    finally:
        logger.info("Session ended: %s", session_id)


async def trio_main():
    logging.basicConfig(level=logging.INFO)
    await trio_websocket.serve_websocket(handler, host=None, port=5000, ssl_context=None)


def main():
    trio_asyncio.run(trio_main)

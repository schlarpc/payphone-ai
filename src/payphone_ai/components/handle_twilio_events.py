import audioop
import base64
import json
import logging

import trio
import trio_websocket

logger = logging.getLogger(__name__)


async def main(
    websocket: trio_websocket.WebSocketConnection,
    human_audio_sender: trio.MemorySendChannel,
) -> None:
    while True:
        try:
            message = await websocket.get_message()
            data = json.loads(message)
            if data["event"] == "media":
                human_audio = audioop.ulaw2lin(base64.b64decode(data["media"]["payload"]), 2)
                await human_audio_sender.send(human_audio)
            elif data["event"] == "closed":
                logger.info("Closed message received: %r", message)
                break
        except trio_websocket.ConnectionClosed:
            logger.info("Connection closed")
            break
    raise Exception("Call ended")

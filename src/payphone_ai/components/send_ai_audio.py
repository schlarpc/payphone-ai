import audioop
import base64
import json

import trio
import trio_websocket


async def main(
    websocket: trio_websocket.WebSocketConnection,
    stream_metadata,
    ai_audio_receiver: trio.MemoryReceiveChannel,
) -> None:
    async with ai_audio_receiver:
        async for ai_audio in ai_audio_receiver:
            await websocket.send_message(
                json.dumps(
                    {
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(audioop.lin2ulaw(ai_audio, 2)).decode(
                                "ascii"
                            )
                        },
                        "streamSid": stream_metadata["streamSid"],
                    }
                )
            )

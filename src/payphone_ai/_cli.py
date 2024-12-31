import logging

import trio
import enum
import signal
import asyncio
import uuid
import random
import sniffio

from vocode.streaming.input_device.base_input_device import BaseInputDevice
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.models.audio_encoding import AudioEncoding


import vocode
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig, ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

class AudioSocketMessageType(enum.IntEnum):
    HANGUP = 0x00
    UUID = 0x01
    AUDIO = 0x10
    ERROR = 0xFF


logger = logging.getLogger()


class AudioSocketInput(BaseInputDevice):
    def __init__(self):
        super().__init__(
            sampling_rate=8000,
            audio_encoding=AudioEncoding.LINEAR16,
            chunk_size=320,
        )

    def get_audio(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

import queue

class AudioSocketOutput(BaseOutputDevice):
    def __init__(self):
        super().__init__(
            sampling_rate=8000,
            audio_encoding=AudioEncoding.LINEAR16,
        )
        self.queue = queue.Queue()

    async def send_async(self, chunk: bytes):
        self.queue.put(chunk)

vocode.setenv(
    OPENAI_API_KEY="",
    DEEPGRAM_API_KEY="",
    ELEVEN_LABS_API_KEY="",
    AZURE_SPEECH_REGION="westus2",
    AZURE_SPEECH_KEY="",
)

async def run_conversation(conversation, audio_input, audio_output):
    await conversation.start()
    while conversation.is_active():
        chunk = audio_input.get_audio()
        if chunk:
            conversation.receive_audio(chunk)
        await asyncio.sleep(0.01)

def run_conversation_sync(conversation, audio_input, audio_output):
    asyncio.run(run_conversation(conversation, audio_input, audio_output))

import time
async def receive_audio(server_stream, audio_input):
    buffer = bytearray()
    async for data in server_stream:
        buffer.extend(data)
        while len(buffer) >= 3:
            payload_type = AudioSocketMessageType(buffer[0])
            payload_length = int.from_bytes(buffer[1:3], 'big')
            if len(buffer) < payload_length + 3:
                break
            payload = bytes(buffer[3:payload_length + 3])
            del buffer[:payload_length + 3]
            match payload_type:
                case AudioSocketMessageType.AUDIO:
                    audio_input.queue.put(payload)
                case AudioSocketMessageType.UUID:
                    logger.info("UUID %s", uuid.UUID(bytes=payload))
                case _:
                    ...

async def send_audio(server_stream, audio_output):
    while True:
        try:
            chunk = bytearray(audio_output.queue.get_nowait())
            print(len(chunk))
            while chunk:
                with open(f"{time.time()}.raw", "wb") as f:
                    f.write(chunk)
                subchunk = chunk[:320]
                del chunk[:320]
                await server_stream.send_all(b'\x10' + len(subchunk).to_bytes(2, 'big') + subchunk)
                await trio.sleep(len(subchunk) / 2 / 8000)
            print('chunk dun')
        except queue.Empty:
            pass
        await trio.sleep(0.01)

import functools
async def echo_server(server_stream):
    audio_input = AudioSocketInput()
    audio_output = AudioSocketOutput()
    conversation = StreamingConversation(
        output_device=audio_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                audio_input, endpointing_config=PunctuationEndpointingConfig()
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hello!"),
                prompt_preamble="Have a pleasant conversation about life",
            ),
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(audio_output)
        ),
    )
    # send a non-empty silent data packet so Asterisk will actually start sending data
    await server_stream.send_all(b'\x10\x00\x02\x00\x00')
    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(receive_audio, server_stream, audio_input)
            nursery.start_soon(send_audio, server_stream, audio_output)
            nursery.start_soon(
                functools.partial(
                    trio.to_thread.run_sync,
                    run_conversation_sync,
                    conversation,
                    audio_input,
                    audio_output,
                    cancellable=True,
                )
            )
    except Exception:
        logger.exception("Call killed")
    conversation.terminate()
    print("Done")


async def trio_main():
    logging.basicConfig(level=logging.INFO)
    await trio.serve_tcp(echo_server, 9093)


def main():
    trio.run(trio_main)

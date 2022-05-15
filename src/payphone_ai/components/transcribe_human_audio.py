import audioop
import contextlib
import ctypes
import functools

import amazon_transcribe.auth
import amazon_transcribe.client
import boto3
import numpy
import trio
from trio_asyncio import asyncio_as_trio as a2t


class Boto3CredentialResolver(amazon_transcribe.auth.CredentialResolver):
    def __init__(self, session=None):
        self._session = session or boto3.Session()

    async def get_credentials(self):
        creds = self._session.get_credentials().get_frozen_credentials()
        return amazon_transcribe.auth.Credentials(
            creds.access_key,
            creds.secret_key,
            creds.token,
        )


@functools.cache
def load_rnnoise_lib():
    lib = ctypes.cdll.LoadLibrary(
        "/nix/store/phg1m5bf4n49msvnrngz35jg6ab1p5xs-rnnoise-2021-01-22/lib/librnnoise.so"
    )
    lib.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rnnoise_process_frame.restype = ctypes.c_float
    lib.rnnoise_create.restype = ctypes.c_void_p
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    return lib


def process_rnnoise_frame(_rnnoise, _instance, data, *, _frame_size=480):
    assert len(data) == _frame_size * 2  # 16-bit samples
    output = numpy.ndarray((_frame_size,), "h", data).astype(ctypes.c_float)
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    vad_prob = _rnnoise.rnnoise_process_frame(_instance, output_ptr, output_ptr)
    return vad_prob, output.astype(ctypes.c_short).tobytes()


@contextlib.contextmanager
def create_rnnoise_instance(*, rnnoise=None):
    if rnnoise is None:
        rnnoise = load_rnnoise_lib()
    instance = rnnoise.rnnoise_create(None)
    try:
        yield functools.partial(process_rnnoise_frame, rnnoise, instance)
    finally:
        rnnoise.rnnoise_destroy(instance)


async def _aws_transcribe_sender(
    human_audio_receiver: trio.MemoryReceiveChannel, transcription_sender
):
    async with human_audio_receiver:
        with create_rnnoise_instance() as do_rnnoise:
            state = None
            buffer = bytearray()
            async for human_audio in human_audio_receiver:
                resampled_human_audio, state = audioop.ratecv(
                    human_audio,
                    2,
                    1,
                    8000,
                    48000,
                    state,
                )
                buffer += resampled_human_audio
                while len(buffer) >= 960:
                    _, denoised_human_audio = do_rnnoise(bytes(buffer[:960]))
                    buffer = buffer[960:]
                    await transcription_sender(denoised_human_audio)


async def _aws_transcribe_receiver(
    transcription_receiver, human_text_sender: trio.MemorySendChannel
):
    async with human_text_sender:
        async for event in transcription_receiver:
            for result in event.transcript.results:
                if result.is_partial:
                    continue
                human_text = result.alternatives[0].transcript.strip()
                await human_text_sender.send(human_text)


async def main(
    human_audio_receiver: trio.MemoryReceiveChannel, human_text_sender: trio.MemorySendChannel
) -> None:
    async with human_text_sender:
        client = amazon_transcribe.client.TranscribeStreamingClient(
            region="us-west-2",
            credential_resolver=Boto3CredentialResolver(),
        )
        transcription = await a2t(client.start_stream_transcription)(
            language_code="en-US",
            media_encoding="pcm",
            media_sample_rate_hz=48000,
        )
        try:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(
                    _aws_transcribe_sender,
                    human_audio_receiver,
                    a2t(transcription.input_stream.send_audio_event),
                )
                nursery.start_soon(
                    _aws_transcribe_receiver,
                    a2t(transcription.output_stream),
                    human_text_sender,
                )
        finally:
            await a2t(transcription.input_stream.end_stream)()

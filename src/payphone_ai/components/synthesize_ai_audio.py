import aiobotocore.session
import trio
from trio_asyncio import asyncio_as_trio as a2t

from payphone_ai.prompts import ResolvedPrompt


async def main(
    prompt: ResolvedPrompt,
    ai_text_receiver: trio.MemoryReceiveChannel,
    ai_audio_sender: trio.MemorySendChannel,
) -> None:
    session = aiobotocore.session.get_session()
    async with a2t(session.create_client("polly")) as polly, ai_text_receiver, ai_audio_sender:
        async for ai_text in ai_text_receiver:
            response = await a2t(
                polly.synthesize_speech(
                    Text=ai_text,
                    VoiceId=prompt.voice_id,
                    Engine=prompt.engine,
                    LanguageCode="en-US",
                    OutputFormat="pcm",
                    SampleRate="8000",
                )
            )
            async with a2t(response["AudioStream"]) as stream:
                while ai_audio := await a2t(stream.content.read)(1024):
                    await ai_audio_sender.send(ai_audio)

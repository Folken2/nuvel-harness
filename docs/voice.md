# Voice

ADK's `CascadeLive` wraps a text model in speech to text on the way in and text to speech on
the way out. Many speech providers only transcribe a whole clip, or speak one text per call.
This package adapts those to ADK's live interfaces: `ClipSTT` finds where each utterance starts
and ends, and `SentenceTTS` speaks a streamed reply sentence by sentence, so audio starts before
the reply ends. Both report what each call cost. A product imports `from harness.voice import
ClipSTT, EndpointSettings, SentenceTTS, UsageTally, ...`.

The provider calls themselves are the product's: it passes them in as callables.

## What it guarantees

### `ClipSTT` (an ADK `LiveIngress`)

- **Audio in.** 16-bit mono PCM blobs. The rate is read from the blob's mime type
  (`audio/pcm;rate=24000`), else the last one seen, starting at `DEFAULT_RATE` (16000).
- **Utterances found by loudness.** With `endpoint=EndpointSettings(...)`, each 20 ms frame's
  loudness is compared with a noise floor that follows the room while nobody speaks. Speech
  louder than `ratio` times the floor, and at least `min_level`, for `start_ms` yields
  `UserSpeechStarted`; quiet for `end_ms` ends the utterance and yields
  `UserTurnFinished(text=...)`. The 500 ms before speech was detected are kept, so the first
  word isn't cut.
- **Without endpointing** (`endpoint=None`, the default), each audio stream it is given is one
  utterance: `UserSpeechStarted` at its first frame, `UserTurnFinished` when it ends.
- **Short or silent clips cost nothing.** A clip under 0.2 seconds is never sent, and an empty
  transcript yields no event.
- **Captions while someone speaks.** Every `partial_every_seconds` of speech (default 1.2), the
  clip so far is transcribed and yielded as `PartialTranscript`, one call at a time. A failed
  partial is logged and dropped, not raised. A partial still running when the utterance ends is
  cancelled. `partial_every_seconds=None` turns captions off.
- **Every call is reported.** Each transcription, partials included, calls `on_usage` with an
  `AudioUsage` of kind `speech_to_text`, the model, the provider's `cost_usd` and the seconds
  heard.

### `SentenceTTS` (an ADK `LiveEgress`)

- **Sentence by sentence.** Text deltas are cut where `.`, `!`, `?`, `:` or `;` is followed by
  space (closing quotes and brackets stay with the sentence), or at a line break. A piece
  shorter than 24 characters waits to join the next one; the rest is spoken when the text ends.
- **Only words are spoken.** Each sentence goes through `speakable.speakable` first: code
  blocks are left out, links and images become their text, bare URLs are dropped, and headings,
  bullets, quotes, rules, table rules and emphasis marks are removed. A sentence with nothing
  left to say is skipped.
- **Order and overlap.** For each sentence it yields its `AudioChunk`s (with the given
  `mime_type`), then `AgentSpokenOutput` with the sentence as written. The next sentence's call
  starts while the current one's audio is still arriving, at most one ahead.
- **Barge-in.** When the `cancel` event is set, output stops at the next chunk and calls in
  flight are cancelled.
- **A provider error is raised**, after the audio that sentence already sent.
- **Every sentence is reported** to `on_usage` as `text_to_speech`, with the characters spoken
  and `price_per_char * characters`. A price of `None` is logged and recorded as zero cost. The
  record is made when the sentence's call starts, so a sentence cut by a barge-in still counts.

### Usage

- `UsageSink` is any `Callable[[AudioUsage], None]`. `UsageTally` is one: it keeps `calls`, and
  gives `cost_usd` (the sum), `amount` (the sum of seconds and characters, so only meaningful
  for one kind) and `model` (the last call's).

## What it costs

- Endpointing: nothing beyond arithmetic on each 20 ms frame. No model, no network.
- Speech to text: one provider call per utterance, plus one per `partial_every_seconds` of
  speech while it lasts. Each partial sends the whole clip so far. The utterance is held in
  memory until it ends, plus the 500 ms pre-roll.
- Text to speech: one `speak` call per sentence, and, when `on_usage` is set, one
  `price_per_char` call per sentence. At most two sentences' calls are in flight.
- Nothing is stored: usage goes only to the sink you pass, and keeping it is the product's
  call.

## Configure it

`EndpointSettings` (defaults as written): `ratio=3.0`, `min_level=0.015` (loudness from 0 to
1), `start_ms=120`, `end_ms=700`.

`ClipSTT(*, key, model, transcribe, language=None, endpoint=None, partial_every_seconds=1.2,
on_usage=None)`. `transcribe` is called as `transcribe(wav_bytes, key=..., model=...,
language=...)` and returns an object with `text`, `cost_usd` and `seconds`.

`SentenceTTS(*, key, model, voice, speak, price_per_char, mime_type, on_usage=None)`. `speak`
is called as `speak(words, key=..., model=..., voice=...)` and yields audio bytes as they
arrive; `price_per_char` is awaited as `price_per_char(model, key=...)` and returns a float or
`None`.

```python
from google.adk.agents import LlmAgent
from google.adk.live import CascadeLive

from harness.voice import ClipSTT, EndpointSettings, SentenceTTS, UsageTally

tally = UsageTally()
stt = ClipSTT(key=stt_key, model="my-stt-model", transcribe=my_transcribe, language="en",
              endpoint=EndpointSettings(end_ms=600), on_usage=tally)
tts = SentenceTTS(key=tts_key, model="my-tts-model", voice="my-voice", speak=my_speak,
                  price_per_char=my_price, mime_type="audio/pcm;rate=24000", on_usage=tally)

agent = LlmAgent(name="voice", model=CascadeLive(model="my-text-model", stt=stt, tts=tts),
                 instruction="...")
# After the request: tally.cost_usd, tally.calls
```

## Test code that uses it

The provider is three callables, so a test passes fakes: an async `transcribe` that returns an
object with `text`, `cost_usd` and `seconds`; an async generator `speak` that yields a few byte
strings; an async `price_per_char` returning a fixed number. Drive `ClipSTT` by calling it with
an async iterator of `google.genai.types.Blob` frames (silence is zero bytes, speech any loud
16-bit samples), and `SentenceTTS` with an async iterator of text deltas and an
`asyncio.Event`. Collect the yielded events and assert on their types, and on a `UsageTally`
passed as `on_usage`. No network, no model. The harness has no test file for this package yet.

## Reference

| Name | What it is |
| --- | --- |
| `ClipSTT` | ADK `LiveIngress` over a whole-clip transcription call, with endpointing and captions. |
| `DEFAULT_RATE` | 16000: the PCM rate assumed until a blob's mime type says otherwise. |
| `EndpointSettings` | Loudness thresholds and timings for finding utterances. |
| `SentenceTTS` | ADK `LiveEgress` that speaks a streamed reply one sentence per call. |
| `UsageSink` | `Callable[[AudioUsage], None]`: where each call's cost goes. |
| `UsageTally` | A sink that adds up one request's calls. |
| `pcm` | Submodule: `pcm.rate_of(mime_type, default)` and `pcm.duration_seconds(pcm, rate)`. |
| `speakable` | Submodule: `speakable.speakable(text)` turns Markdown into the words to say. |

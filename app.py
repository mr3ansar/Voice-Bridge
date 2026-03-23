## Speech-to-Speech Streamlit App — Full Assignment Solution
## Features:
##   - Auto-run: pipeline runs as soon as audio is recorded (no button needed)
##   - Chat Bubbles: st.chat_message() for professional UI
##   - Clear Chat: session_state reset button
##   - Latency Timer: ASR / LLM / TTS time measured and displayed
##   - Voice Switch: gTTS language dropdown + multiple Piper voices selectable

from __future__ import annotations
import os
import time
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    # ASR
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int

    # LLM
    groq_api_key: str
    groq_model_id: str
    system_prompt: str

    # TTS
    tts_engine: str
    piper_model_path: str
    piper_config_path: str


def load_config() -> AppConfig:
    def _get_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    return AppConfig(
        whisper_model=os.getenv("WHISPER_MODEL", "base.en").strip(),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en").strip(),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip(),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip(),
        whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", 4),

        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant").strip(),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful voice assistant for students. Keep replies short and clear."
        ).strip(),

        tts_engine=os.getenv("TTS_ENGINE", "gtts").strip().lower(),
        piper_model_path=os.getenv("PIPER_MODEL_PATH", "").strip(),
        piper_config_path=os.getenv("PIPER_CONFIG_PATH", "").strip(),
    )


CFG = load_config()


# ─────────────────────────────────────────────
# ASR — Faster Whisper
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size: str, device: str, compute_type: str, cpu_threads: int):
    """
    @st.cache_resource: model sirf ek baar load hoga, Streamlit reruns pe dobara nahi.
    Low RAM tip: model_size = tiny.en, compute_type = int8
    """
    from faster_whisper import WhisperModel
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    """
    WAV bytes ko temp file mein save karo, phir whisper se transcribe karo.
    finally block: temp file hamesha delete hogi, chahe error aaye ya na aaye.
    """
    model = get_whisper_model(
        CFG.whisper_model,
        CFG.whisper_device,
        CFG.whisper_compute_type,
        CFG.whisper_cpu_threads,
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        segments, _info = model.transcribe(
            tmp_path,
            language=CFG.whisper_language,
            beam_size=1,
            vad_filter=True,   # silence / noise ignore karo
        )
        return "".join(seg.text for seg in segments).strip()

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────
# LLM — Groq
# ─────────────────────────────────────────────
def offline_demo_reply(user_text: str) -> str:
    if not user_text:
        return "I did not catch that. Please try again."
    return (
        "Offline demo mode: I can hear you, but I am not connected to an AI model yet.\n\n"
        f"You said: {user_text}\n\n"
        "To enable real AI replies, add GROQ_API_KEY in your .env."
    )


def groq_chat_completion(messages: List[Dict[str, str]]) -> str:
    import requests

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {CFG.groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": CFG.groq_model_id,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 250,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def generate_reply(user_text: str, history: List[Dict[str, str]]) -> str:
    if not CFG.groq_api_key:
        return offline_demo_reply(user_text)

    trimmed = history[-6:] if len(history) > 6 else history

    # Sirf role aur content bhejo — audio bytes JSON serializable nahi hain
    clean = [{"role": m["role"], "content": m["content"]} for m in trimmed]

    messages = [{"role": "system", "content": CFG.system_prompt}]
    messages.extend(clean)
    messages.append({"role": "user", "content": user_text})

    try:
        return groq_chat_completion(messages)
    except Exception as e:
        return f"Could not reach Groq. Error: {e}"


# ─────────────────────────────────────────────
# TTS — gTTS (online) or Piper (offline)
# ─────────────────────────────────────────────

# gTTS voices dropdown ke liye — naam: lang code
GTTS_VOICES: Dict[str, str] = {
    "English (US)"  : "en",
    "English (UK)"  : "en-gb",
    "Urdu"          : "ur",
    "Hindi"         : "hi",
    "French"        : "fr",
    "Spanish"       : "es",
}

# Piper voices dropdown ke liye — naam: (model_path, config_path)
# Apni downloaded voices yahan add karo
# Har entry mein: "Display Name": ("path/to/model.onnx", "path/to/model.onnx.json")
PIPER_VOICES: Dict[str, Tuple[str, str]] = {
    "English — Amy (Female)" : (
        "voices/en_US-amy-low.onnx",
        "voices/en_US-amy-low.onnx.json",
    ),
    "English — Ryan (Male)"     : (
        "voices/en_US-ryan-medium.onnx",
        "voices/en_US-ryan-medium.onnx.json",
    ),
    "English — Bryce (Male)"        : (
        "voices/en_US-bryce-medium.onnx",
        "voices/en_US-bryce-medium.onnx.json",
    ),
}


@st.cache_resource(show_spinner=False)
def get_piper_voice(model_path: str, config_path: str):
    """
    Piper voice ek baar load karo aur cache karo.
    Agar user voice switch kare toh naya load hoga, warna cached wala use hoga.
    model_path aur config_path cache key bante hain — same path = same cached voice.
    """
    from piper import PiperVoice
    return PiperVoice.load(model_path, config_path)


def tts_to_audio_file(
    text: str,
    gtts_lang: str = "en",
    piper_model: str = "",
    piper_config: str = "",
) -> Tuple[bytes, str, str]:
    """
    Returns: (audio_bytes, mime_type, file_name)
    gtts_lang    : gTTS ke liye sidebar dropdown se aata hai
    piper_model  : Piper ke liye selected .onnx path
    piper_config : Piper ke liye selected .onnx.json path
    """
    text = (text or "").strip() or "I do not have a response."

    if CFG.tts_engine == "piper":
        return _piper_tts(text, piper_model, piper_config)
    return _gtts_tts(text, gtts_lang)


def _gtts_tts(text: str, lang: str) -> Tuple[bytes, str, str]:
    from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        out_path = f.name
    try:
        gTTS(text=text, lang=lang).save(out_path)
        with open(out_path, "rb") as rf:
            return rf.read(), "audio/mpeg", "reply.mp3"
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


def _piper_tts(text: str, model_path: str, config_path: str) -> Tuple[bytes, str, str]:
    import wave

    # Agar paths empty hain toh gTTS pe fallback karo
    if not model_path or not config_path:
        return _gtts_tts(
            "Piper voice not configured. Please select a valid voice.",
            lang="en",
        )

    # Cache se voice lo — same path pe dobara load nahi hoga
    voice = get_piper_voice(model_path, config_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    try:
        with wave.open(out_path, "wb") as wf:
            voice.synthesize_wav(text, wf)
        with open(out_path, "rb") as rf:
            return rf.read(), "audio/wav", "reply.wav"
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="VoiceBridge", page_icon="🎙️", layout="centered")
st.title("🎙️ VoiceBridge — Speech to Speech AI")

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Settings")

    # TTS engine ke hisaab se alag dropdown dikhao
    if CFG.tts_engine == "piper":
        # Piper mode: available voices ki list dikhao
        st.write("🎙️ Piper Voice")

        # PIPER_VOICES dict mein se sirf woh voices dikhao jinki files exist karti hain
        available_piper = {
            name: paths
            for name, paths in PIPER_VOICES.items()
            if os.path.exists(paths[0]) and os.path.exists(paths[1])
        }

        if not available_piper:
            # Koi bhi voice file nahi mili
            st.warning(
                "No Piper voice files found. "
                "Download .onnx + .onnx.json files into your voices/ folder "
                "and add them to PIPER_VOICES dict in app.py"
            )
            selected_piper_model = ""
            selected_piper_config = ""
        else:
            selected_voice_name = st.selectbox(
                "Select voice",
                options=list(available_piper.keys()),
            )
            # Chunni hui voice ke paths nikalo
            selected_piper_model, selected_piper_config = available_piper[selected_voice_name]
            st.caption(f"Model: `{selected_piper_model}`")

        selected_gtts_lang = "en"  # Piper mode mein gTTS lang ki zaroorat nahi

    else:
        # gTTS mode: language dropdown dikhao
        selected_voice_label = st.selectbox(
            "🔊 TTS Language (gTTS)",
            options=list(GTTS_VOICES.keys()),
            index=0,
        )
        selected_gtts_lang = GTTS_VOICES[selected_voice_label]
        selected_piper_model = ""
        selected_piper_config = ""

    st.divider()
    st.write("Current config (.env):")
    st.code(
        "\n".join([
            f"WHISPER_MODEL       = {CFG.whisper_model}",
            f"WHISPER_LANGUAGE    = {CFG.whisper_language}",
            f"WHISPER_DEVICE      = {CFG.whisper_device}",
            f"WHISPER_COMPUTE_TYPE= {CFG.whisper_compute_type}",
            f"WHISPER_CPU_THREADS = {CFG.whisper_cpu_threads}",
            f"GROQ_API_KEY        = {'SET ✅' if CFG.groq_api_key else 'NOT SET ❌'}",
            f"GROQ_MODEL_ID       = {CFG.groq_model_id}",
            f"TTS_ENGINE          = {CFG.tts_engine}",
        ])
    )
    st.caption("💡 Low RAM? Use whisper tiny.en + int8")

# ── Session State Setup ─────────────────────────
# Pehli baar session shuru ho toh yeh sab initialize karo
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_audio_id" not in st.session_state:
    # ASSIGNMENT: Auto-run ke liye — last processed audio ka hash store karo
    st.session_state.last_audio_id = None

if "latency" not in st.session_state:
    # ASSIGNMENT: Latency store karne ke liye
    st.session_state.latency = {}

# ── ASSIGNMENT: Clear Chat Button ───────────────
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑️ Clear", help="Clear all chat history"):
        st.session_state.chat_history = []
        st.session_state.last_audio_id = None
        st.session_state.latency = {}
        st.rerun()  # page ko refresh karo taake UI bhi reset ho

# ── ASSIGNMENT: Chat Bubbles ─────────────────────
# Purani conversation display karo
st.write("### Conversation")
for msg in st.session_state.chat_history:
    # st.chat_message("user") -> right-aligned human bubble
    # st.chat_message("assistant") -> left-aligned AI bubble
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Agar is message ke saath audio saved hai
        if "audio" in msg:
            st.audio(msg["audio"]["bytes"], format=msg["audio"]["mime"])

# ── Audio Input ─────────────────────────────────
st.divider()
audio_value = st.audio_input("🎤 Record your voice")

if audio_value:
    wav_bytes = audio_value.getvalue()

    # ASSIGNMENT: Auto-run logic
    # hash() se audio ka unique ID banao
    # Agar yeh audio pehle process ho chuka hai, dobara mat chalao
    current_audio_id = hash(wav_bytes)

    if current_audio_id != st.session_state.last_audio_id:
        # Naya audio mila! Pipeline chalao
        st.session_state.last_audio_id = current_audio_id

        # ── Step 1: ASR ──────────────────────────
        with st.spinner("🎧 Transcribing your speech..."):
            asr_start = time.time()                         # timer shuru
            transcript = transcribe_wav_bytes(wav_bytes)
            asr_time = time.time() - asr_start              # timer khatam

        if not transcript:
            st.error("❌ No speech detected. Try in a quieter environment.")
            st.stop()

        # ── Step 2: LLM ──────────────────────────
        with st.spinner("🤖 Generating reply..."):
            llm_start = time.time()
            reply_text = generate_reply(transcript, st.session_state.chat_history)
            llm_time = time.time() - llm_start

        # ── Step 3: TTS ──────────────────────────
        with st.spinner("🔊 Converting reply to speech..."):
            tts_start = time.time()
            audio_bytes, mime, fname = tts_to_audio_file(
                reply_text,
                gtts_lang=selected_gtts_lang,
                piper_model=selected_piper_model,
                piper_config=selected_piper_config,
            )
            tts_time = time.time() - tts_start

        # ── Save latency to session ───────────────
        # ASSIGNMENT: Latency store karo
        st.session_state.latency = {
            "asr": asr_time,
            "llm": llm_time,
            "tts": tts_time,
            "total": asr_time + llm_time + tts_time,
        }

        # ── Update chat history ───────────────────
        st.session_state.chat_history.append({
            "role": "user",
            "content": transcript,
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply_text,
            "audio": {"bytes": audio_bytes, "mime": mime},
        })

        # ── Rerun so chat bubbles update ──────────
        # st.rerun() page dobara chalata hai
        # ab chat_history mein naye messages hain toh bubbles show honge
        st.rerun()

# ── ASSIGNMENT: Latency Display ─────────────────
if st.session_state.latency:
    lat = st.session_state.latency
    st.divider()
    st.write("#### ⏱️ Last Request Latency")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ASR",   f"{lat['asr']:.2f}s")
    c2.metric("LLM",   f"{lat['llm']:.2f}s")
    c3.metric("TTS",   f"{lat['tts']:.2f}s")
    c4.metric("Total", f"{lat['total']:.2f}s")

# ── Debug Section ────────────────────────────────
st.divider()
if st.checkbox("🐛 Show raw chat history (debug)"):
    # Audio bytes show nahi karte (bohat bari hoti hain)
    debug_history = [
        {k: v for k, v in msg.items() if k != "audio"}
        for msg in st.session_state.chat_history
    ]
    st.json(debug_history)
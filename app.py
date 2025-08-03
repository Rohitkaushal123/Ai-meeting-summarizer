import os
import tempfile

# —————— 0. Vendor your local FFmpeg build ——————
ffmpeg_bin = os.path.join(
    os.path.dirname(__file__),
    "ffmpeg-2025-07-31-git-119d127d05-essentials_build",
    "bin"
)
os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

# —————— 1. Standard imports ——————
import whisper
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit as st
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# —————— 2. Load environment & init Groq model ——————
load_dotenv()
groq_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# —————— 3. Whisper & HF ASR loaders ——————
@st.cache_resource
def get_whisper_model(size: str = "small"):
    return whisper.load_model(size)

@st.cache_resource
def get_hf_asr(model_name: str = "facebook/wav2vec2-base-960h"):
    return pipeline("automatic-speech-recognition", model=model_name)

whisper_model = get_whisper_model()
hf_asr = get_hf_asr()

# —————— 4. Transcription functions ——————
def transcribe_whisper(audio_bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes.read()); tmp.flush(); tmp.close()
    text = whisper_model.transcribe(tmp.name)["text"]
    os.unlink(tmp.name)
    return text

def transcribe_hf(audio_bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes.read()); tmp.flush(); tmp.close()
    text = hf_asr(tmp.name)["text"]
    os.unlink(tmp.name)
    return text

# —————— 5. Prompt templates & analysis ——————
summary_tpl = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are an expert meeting assistant. Summarize the following meeting transcript in clear bullet points:

{transcript}

Summary:
"""
)
decisions_tpl = PromptTemplate(
    input_variables=["transcript"],
    template="""
Read the following meeting transcript and extract all key decisions made during the meeting.

{transcript}

Key Decisions:
"""
)
actions_tpl = PromptTemplate(
    input_variables=["transcript"],
    template="""
From the meeting transcript below, extract all action items. Specify WHO needs to do WHAT, and by WHEN (if mentioned).

{transcript}

Action Items:
"""
)


def analyze_meeting(transcript: str):
    raw_summary   = groq_model.invoke(summary_tpl.format(transcript=transcript))
    raw_decisions = groq_model.invoke(decisions_tpl.format(transcript=transcript))
    raw_actions   = groq_model.invoke(actions_tpl.format(transcript=transcript))

    # AIMessage.content holds the text
    summary   = raw_summary.content.strip()
    decisions = raw_decisions.content.strip()
    actions   = raw_actions.content.strip()
    return summary, decisions, actions

# —————— 6. Streamlit UI ——————
st.set_page_config(page_title="MinuteMate Plus ASR", layout="wide")
st.title("🕒 MinuteMate — Meeting Analysis App")

# Model choice
st.subheader("Choose ASR Model")
asr_choice = st.radio( "Select Model",
                       ["🟢 **Whisper (OpenAI):** High accuracy, supports many languages, punctuation & casing, but **slower** inference.", 
                       "🔴 **Wav2Vec2:** Very fast & lightweight, but slightly lower accuracy, lacks punctuation/casing."])





text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap = 200)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

manual_transcript = st.text_area("Copy and paste the transcript here if you want to analyze it without uploading an audio file:")

if st.button("Analyze Transcript"):
    if manual_transcript.strip() == "":
        st.warning("Please paste a transcript to analyze.")
    else:
        transcript = manual_transcript
        st.info("You are using the manual transcript input. Please ensure the transcript is accurate.")

        if len(transcript) > 1000:
            chunks = text_splitter.split_text(transcript)
            embeddings = embedding.embed_documents(chunks)
            # st.write(f"Text split into {len(chunks)} chunks for embedding.")
        else:
            embeddings = embedding.embed_documents([transcript])
            st.write("Transcript is short, embedding whole text.")
        summary, decisions, actions = analyze_meeting(transcript)

        st.subheader("🗒️ Summary")
        st.write(summary)
        st.subheader("⚖️ Key Decisions")
        st.write(decisions)
        st.subheader("✅ Action Items")
        st.write(actions)


# File uploader
uploaded_file = st.file_uploader("Choose audio file", type=["mp3", "wav", "m4a"])
if not uploaded_file:
    st.info("Please upload an audio file.")
    st.stop()

st.write(f"Transcribing with **{asr_choice}**...")

if asr_choice.startswith("Whisper"):
    transcript = transcribe_whisper(uploaded_file)
else:
    transcript = transcribe_hf(uploaded_file)

st.success("Transcription complete")

if len(transcript) > 1000:
    chunks = text_splitter.split_text(transcript)
    embeddings = embedding.embed_documents(chunks)
    st.write(f"Text split into {len(chunks)} chunks for embedding.")
else:
    embeddings = embedding.embed_documents([transcript])
    st.write("Transcript is short, embedding whole text.")

summary, decisions, actions = analyze_meeting(transcript)

st.subheader("🗒️ Summary")
st.write(summary)
st.subheader("⚖️ Key Decisions")
st.write(decisions)
st.subheader("✅ Action Items")
st.write(actions)






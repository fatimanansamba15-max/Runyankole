import torch
import streamlit as st
import pandas as pd
import os
import string
import io
from gtts import gTTS
# NEW OFFLINE PIPELINE: HuggingFace tools to run AI speech recognition on your computer
from transformers import pipeline
import soundfile as sf


@st.cache_resource
def load_speech_model():
    # Loads OpenAI's lightweight, highly accurate English voice model locally
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")


@st.cache_data
def load_dict():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'runyankolee.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['english', 'runyankole'])

    df['english'] = df['english'].str.strip().str.lower()
    df['runyankole'] = df['runyankole'].str.strip().str.lower()
    return df


df = load_dict()
mapping_dict = dict(zip(df['english'], df['runyankole']))


def apply_runyankole_grammar(words, selected_tense_mode):
    processed_words = []

    hardcoded_phrases = {
        ("good", "morning"): "agandi",
        ("how", "are", "you"): "agandi",
        ("thank", "you"): "webare",
        ("thank", "you", "very", "much"): "webare munonga",
        ("praise", "god"): "mukama asiimwe",
        ("and", "you"): "niiwe",
    }

    if selected_tense_mode == "Auto-Detect Tense":
        sentence_tense = "present"
        if any(w in words for w in ["will", "tomorrow", "later"]):
            sentence_tense = "future"
        elif any(w in words for w in ["did", "was", "were", "yesterday", "ago", "went"]):
            sentence_tense = "past"
    else:
        sentence_tense = selected_tense_mode.lower()

    cleaned_words = [w for w in words if w not in ["will", "did", "am", "is", "are", "was", "were"]]
    i = 0
    n = len(cleaned_words)

    while i < n:
        found_csv_phrase = False
        for phrase_length in (4, 3, 2):
            if i + phrase_length <= n:
                word_chunk = tuple(cleaned_words[i:i + phrase_length])
                phrase_string = " ".join(word_chunk)

                if word_chunk in hardcoded_phrases:
                    processed_words.append(hardcoded_phrases[word_chunk])
                    i += phrase_length
                    found_csv_phrase = True
                    break
                elif phrase_string in mapping_dict:
                    processed_words.append(mapping_dict[phrase_string])
                    i += phrase_length
                    found_csv_phrase = True
                    break

        if found_csv_phrase:
            continue

        possessives = ["my", "your", "his", "her", "our", "their"]
        if cleaned_words[i] in possessives and i + 1 < n:
            possessive_word = cleaned_words[i]
            noun_word = cleaned_words[i + 1]
            runya_noun = mapping_dict.get(noun_word, f"[{noun_word}]")
            runya_poss = mapping_dict.get(possessive_word, f"[{possessive_word}]")
            processed_words.append(f"{runya_noun} {runya_poss}")
            i += 2
            continue

        subjects = {"i": "n", "you": "o", "he": "a", "she": "a", "we": "tu", "they": "ba"}
        if cleaned_words[i] in subjects and i + 1 < n:
            subject = cleaned_words[i]
            verb_word = cleaned_words[i + 1]
            runya_verb = mapping_dict.get(verb_word, None)
            if runya_verb:
                if runya_verb.startswith("ku") and len(runya_verb) > 2:
                    verb_root = runya_verb[2:]
                elif runya_verb.startswith("kw") and len(runya_verb) > 2:
                    verb_root = runya_verb[2:]
                else:
                    verb_root = runya_verb

                prefix = subjects[subject]

                if sentence_tense == "future":
                    if prefix == "n":
                        conjugated_verb = f"ndyaa{verb_root}"
                    else:
                        conjugated_verb = f"{prefix}ryaa{verb_root}"
                elif sentence_tense == "past":
                    if prefix == "n":
                        conjugated_verb = f"naka{verb_root}"
                    else:
                        conjugated_verb = f"{prefix}ka{verb_root}"
                else:
                    if prefix == "n":
                        conjugated_verb = f"nin{verb_root}"
                    elif prefix == "o":
                        conjugated_verb = f"noo{verb_root}"
                    else:
                        conjugated_verb = f"{prefix}nee{verb_root}"

                processed_words.append(conjugated_verb)
                i += 2
                continue

        processed_words.append(mapping_dict.get(cleaned_words[i], f"[{cleaned_words[i]}]"))
        i += 1

    return " ".join(processed_words)


# --- Streamlit Layout Configuration ---
st.set_page_config(page_title="Runyankole Translator", layout="centered")

st.title("Smart Runyankole Translator")
st.write("Translates complete sentences or words using structural Bantu grammar rules.")

# Sidebar Controls
st.sidebar.header("Translation Settings")
tense_mode = st.sidebar.selectbox(
    "Select Tense Mode:",
    ["Auto-Detect Tense", "Present", "Past", "Future"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Auto-Detect** reads keywords like 'tomorrow' or 'yesterday' to decide verb shapes. "
    "Switch manually to force a specific tense form."
)

# --- OFFLINE AUDIO PROCESSING SECTION ---
if "spoken_sentence" not in st.session_state:
    st.session_state.spoken_sentence = ""

st.write("🎙️ **Voice Input:** Record your voice using the tool below.")
recorded_audio = st.audio_input("Record your voice")

if recorded_audio is not None:
    # Avoid looping on already processed audio buffers
    if "last_processed_audio" not in st.session_state or st.session_state.last_processed_audio != recorded_audio.getvalue():
        with st.spinner("Decoding your voice locally (No internet needed)..."):
            try:
                # Convert the audio buffer byte stream into a format python can analyze
                audio_data, sample_rate = sf.read(io.BytesIO(recorded_audio.read()))

                # If audio records multi-channel, convert to mono channel for the AI engine
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]

                # Process audio locally using Hugging Face Pipeline
                asr_pipeline = load_speech_model()
                result = asr_pipeline({"raw": audio_data, "sampling_rate": sample_rate})

                # Update persistent state memory
                st.session_state.spoken_sentence = result["text"]
                st.session_state.last_processed_audio = recorded_audio.getvalue()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Processing Error: {str(e)}")

# Text input seamlessly synchronized with local AI transcriber engine
sentence = st.text_input(
    "Enter English sentence",
    value=st.session_state.spoken_sentence,
    key="text_entry_field"
)

if sentence:
    cleaned_sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_sentence.split()

    if words:
        st.cache_data.clear()
        translated_result = apply_runyankole_grammar(words, tense_mode)

        # Display the text output card
        st.success(translated_result.capitalize())

        # --- AUDIO GENERATION BLOCK ---
        try:
            clean_speech_text = translated_result.replace('[', '').replace(']', '')
            tts = gTTS(text=clean_speech_text, lang='sw')

            sound_buffer = io.BytesIO()
            tts.write_to_fp(sound_buffer)
            sound_buffer.seek(0)

            st.write("🔊 **Listen to Pronunciation:**")
            st.audio(sound_buffer.read(), format="audio/mp3")

        except Exception as e:
            st.warning("⚠️ Could not generate audio playback stream at this moment.")

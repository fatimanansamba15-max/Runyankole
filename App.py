import streamlit as st
import pandas as pd
import os
import string


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
        ("Praise","God"):"mukama asiimwe",
        ("and","you"):"niiwe",
    
    }

    # 1. PHASE 1: Detect Tense First
    if selected_tense_mode == "Auto-Detect Tense":
        sentence_tense = "present"
        if any(w in words for w in ["will", "tomorrow", "later"]):
            sentence_tense = "future"
        elif any(w in words for w in ["did", "was", "were", "yesterday", "ago", "went"]):
            sentence_tense = "past"
    else:
        sentence_tense = selected_tense_mode.lower()

    # 2. PHASE 2: Clean up English helper words so Subject and Verb sit side-by-side
    cleaned_words = [w for w in words if w not in ["will", "did", "am", "is", "are", "was", "were"]]

    i = 0
    n = len(cleaned_words)

    # 3. PHASE 3: Run Grammar Sequencing
    while i < n:
        # Check CSV and hardcoded phrases first
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

        # Invert English Possessive Adjectives (e.g., "my book" -> "ekitabo wangye")
        possessives = ["my", "your", "his", "her", "our", "their"]
        if cleaned_words[i] in possessives and i + 1 < n:
            possessive_word = cleaned_words[i]
            noun_word = cleaned_words[i + 1]

            runya_noun = mapping_dict.get(noun_word, f"[{noun_word}]")
            runya_poss = mapping_dict.get(possessive_word, f"[{possessive_word}]")

            processed_words.append(f"{runya_noun} {runya_poss}")
            i += 2
            continue

        # Tense Conjugation Engine for Subject + Verb
        subjects = {"i": "n", "you": "o", "he": "a", "she": "a", "we": "tu", "they": "ba"}
        if cleaned_words[i] in subjects and i + 1 < n:
            subject = cleaned_words[i]
            verb_word = cleaned_words[i + 1]

            runya_verb = mapping_dict.get(verb_word, None)
            if runya_verb:
                # Strip base infinitive markers to extract verb root
                if runya_verb.startswith("ku") and len(runya_verb) > 2:
                    verb_root = runya_verb[2:]
                elif runya_verb.startswith("kw") and len(runya_verb) > 2:
                    verb_root = runya_verb[2:]
                else:
                    verb_root = runya_verb

                prefix = subjects[subject]

                if sentence_tense == "future":
                    if prefix == "n":
                        conjugated_verb = f"ndyaa{verb_root}"  # 'nr' phonetically builds 'nd'
                    else:
                        conjugated_verb = f"{prefix}ryaa{verb_root}"

                elif sentence_tense == "past":
                    if prefix == "n":
                        conjugated_verb = f"naka{verb_root}"
                    else:
                        conjugated_verb = f"{prefix}ka{verb_root}"

                else:  # Present
                    if prefix == "n":
                        conjugated_verb = f"nin{verb_root}"
                    elif prefix == "o":
                        conjugated_verb = f"noo{verb_root}"
                    else:
                        conjugated_verb = f"{prefix}nee{verb_root}"

                processed_words.append(conjugated_verb)
                i += 2
                continue

        # Default fallback
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

sentence = st.text_input("Enter English sentence")

if sentence:
    cleaned_sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_sentence.split()

    if words:
        # Clear Streamlit cache if user switches settings to guarantee layout updates
        st.cache_data.clear()

        translated_result = apply_runyankole_grammar(words, tense_mode)
        st.success(translated_result.capitalize())

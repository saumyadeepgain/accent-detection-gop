# Hearing Where Someone Comes From

**How a computer learns to hear an Indian-English accent, and how much of the mother tongue is in it**

This repository contains the end-to-end pipeline and generated notebooks for detecting Indian-English accents and measuring mother-tongue phonetic influence.

> [!NOTE]
> The project is built from scratch and explained from scratch. The code here generates the full `accent_gop_notebook.ipynb` which demonstrates the pipeline.

---

## 🏗️ Repository Structure

- `accent_gop_notebook.ipynb`: The generated, complete end-to-end trace notebook.
- `scripts/`: Contains the modular builder scripts (`build_nb_sec_*.py`) that sequentially assemble the final notebook.
  - *To rebuild the notebook, run the scripts in alphabetical order from the root directory.*

---

## 📖 The Journey, in Six Parts

### 1. What Sound Is
Sound is air being pushed back and forth. A microphone turns these physical pushes into voltage, and then into numbers.
* **Frequencies**: Any wiggly wave is a sum of smooth waves at different speeds (Fourier's idea).
* **Spectrograms**: Stacking these frequency recipes over time gives us a picture of speech. Everything the models look at is a version of this picture.

### 2. What Makes an Accent
Accent lives in the habits a first language leaves behind in a second one:
* **Sound Inventory**: Sounds your first language lacks get replaced by the nearest one you have (e.g., Telugu/Tamil have no 'f', leading to 'fan' → 'phan').
* **Vowel Map**: Reusing the vowel tongue positions you grew up with.
* **Voicing Habits**: Whether vocal folds buzz during certain consonants.
* **Rhythm and Melody**: Syllable timing and pitch variation.

> [!TIP]
> None of these is a mistake. They are the fingerprints of the first language. Our job is to read them, not grade them.

### 3. Turning Sound Into Features
To capture these habits, we initially derived standard features:
* **Pitch (Source)**: Median, range, and movement in semitones (captures melody).
* **Formants (Filter)**: F1/F2 medians over all vowels (captures the vowel map).
* **MFCC (Shape)**: Derived step-by-step (mel filters → log → cosine transform) to capture the overall spectral shape.

### 4. Why the Obvious Way Failed
Training a standard classifier on these features yielded four major traps:
1. **Memorising speakers**: The model learned individual voices (pitch, room, microphone) rather than the accent. *Fix: Split by person, never by clip.*
2. **Hearing the microphone**: Anything that differs between recording sessions (microphones, sex ratios) acts as a shortcut. 
3. **Reading aloud vs. talking**: A model trained on free conversation collapses when tested on a fixed reading passage.
4. **Being sure vs. being right**: The model was often confidently wrong. *Fix: Apply temperature scaling (divide raw scores by a constant) for honest probabilities.*

### 5. The Approach That Worked
Instead of teaching from 13 numbers, we started from a model trained on 94,000 hours of speech: **WavLM**.
* **Borrowed Ears**: A network that already 'knows' how speech is built. We read its middle layers (layers 6 and 9) without retraining it.
* **Language-ID Features**: We added outputs from a Voxlingua107 language-ID model, since its ability to tell Tamil from Hindi leaks useful cues.
* **Deliberately Small Classifier**: We used PCA and a simple logistic regression on top. With ~1,800 parameters, it can only learn broad, repeatable patterns.
* **Honest Scoring**: Long recordings are scored in sections, and we measure the agreement. Below 60% agreement, the verdict is marked as "contested".

### 6. How Much Mother Tongue?
Identifying *which* language is a classifier task. Measuring *how much* influence is present is a grading task (**Goodness of Pronunciation**).
* **Phoneme Recogniser**: wav2vec2 trained on 60 languages' phonemes outputs frame-by-frame probabilities.
* **Forced Alignment**: Lines up the known sentence with the audio.
* **Target vs. Rival**: We grade 7 specific contrasts chosen by what Indian languages actually do to English (e.g., `/s/ vs 'sh'`, `/g/ vs /k/`).
* **Honest Baselines**: Scores are standardised and compared against reference baselines to output a meaningful band (minimal, mild, moderate, strong).

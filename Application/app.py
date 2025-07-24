# ==============================================================================
#  Flask Web Application: Neurodiversity-Inspired Coaching Tool (Voice Input)
# ==============================================================================
# This single file contains a complete Flask application that demonstrates the
# full end-to-end pipeline: Audio File Upload -> Transcription -> Analysis -> Personalization.
#
# To Run This Code:
# 1. Save it as a Python file (e.g., `app.py`).
# 2. Install the necessary libraries:
#    pip install Flask spacy scikit-learn numpy textblob pydub
# 3. Download the spaCy model:
#    python -m spacy download en_core_web_md
# 4. **IMPORTANT**: You will also need to install ffmpeg.
#    - On Mac (using Homebrew): brew install ffmpeg
#    - On Windows: Download from https://ffmpeg.org/download.html and add to your system's PATH.
#    - On Linux (Debian/Ubuntu): sudo apt-get install ffmpeg
# 5. Run the app from your terminal:
#    flask --app app run
# 6. Open your web browser and go to http://127.0.0.1:5000
# ==============================================================================

# --- Core Imports ---
from flask import Flask, render_template_string, request, flash, redirect, url_for
import spacy
from spacy.tokens import Doc
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from textblob import TextBlob
import os
import io
from pydub import AudioSegment
import speech_recognition as sr
from werkzeug.utils import secure_filename

# ==============================================================================
#  1. INITIALIZE FLASK APP AND LOAD NLP MODELS
# ==============================================================================
app = Flask(__name__)
# A secret key is needed for flashing messages
app.config['SECRET_KEY'] = 'your_secret_key_here'
# Create a temporary folder for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- Load spaCy Model ---
print("📚 Loading spaCy medium model...")
nlp = spacy.load("en_core_web_md")
print("✅ spaCy model loaded.")

# ==============================================================================
#  2. THE CORE AI PIPELINE (Functions from previous steps)
# ==============================================================================

# --- The "Golden" Advice Library (Private Knowledge Base) ---
GOLDEN_ADVICE_LIBRARY = [
    {"advice": "player is unfocused and not following the game plan", "concept": "Improve tactical discipline and adhere to the team strategy."},
    {"advice": "player is holding onto the ball or possession for too long", "concept": "Increase speed of play; focus on quicker decisions and team integration."},
    {"advice": "player is making poor decisions and forcing plays that are not there", "concept": "Improve situational awareness and shot/pass selection."},
    {"advice": "player is not being aggressive enough and is playing too passively", "concept": "Increase attacking intent; take calculated risks to pressure opponents."},
    {"advice": "player is being too aggressive and committing unnecessary fouls or errors", "concept": "Channel aggression effectively; play with more control and composure."},
    {"advice": "player's defensive stance is too high and off-balance", "concept": "Maintain a lower center of gravity and a balanced athletic stance."},
]

# --- Annotated Dataset for the Intent Classifier ---
ANNOTATED_DATA = [
    ("Just play the simple pass.", "Actionable"),
    ("It would be brilliant if you could treat your bat less like a shield.", "Actionable"),
    ("Remember to keep your shoulders square to the target.", "Actionable"),
    ("Stop shooting from so far out and look for the open pass instead.", "Actionable"),
    ("Your teammates might as well be selling popcorn in the stands.", "Actionable"),
    ("you're trying to thread a needle through three defenders every time.", "Actionable"),
    ("Honestly, your determination to defend every single ball is admirable, truly.", "Non-Actionable"),
    ("That was a fantastic effort.", "Non-Actionable"),
    ("Look mate, I know you can ping it 40 yards, that's top drawer.", "Non-Actionable")
]

# --- Semantic Searcher Class ---
class SemanticSearcher:
    def __init__(self, knowledge_base: list):
        self.knowledge_base = knowledge_base
        self.kb_vectors = [nlp(entry["advice"]).vector for entry in knowledge_base]

    def _cosine_similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0: return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_closest_advice(self, query_text: str, confidence_threshold=0.5) -> str:
        if not self.knowledge_base: return "Knowledge base is empty."
        query_vector = nlp(query_text).vector
        if np.linalg.norm(query_vector) == 0: return f"Novel advice detected: \"{query_text}\""
        similarities = [self._cosine_similarity(query_vector, kb_vector) for kb_vector in self.kb_vectors]
        best_match_index = np.argmax(similarities)
        best_score = similarities[best_match_index]
        if best_score >= confidence_threshold:
            return self.knowledge_base[best_match_index]["concept"]
        else:
            return f"Novel advice detected: \"{query_text}\""

# --- Intent Classifier Functions ---
def extract_linguistic_features(text: str) -> dict:
    blob = TextBlob(text)
    doc = nlp(text)
    root_verb = ""
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token.lemma_
            break
    features = {
        "sentiment_polarity": blob.sentiment.polarity, "sentiment_subjectivity": blob.sentiment.subjectivity,
        "has_contrast_word": 1 if any(t.lower_ in ["but", "however"] for t in doc) else 0,
        "is_imperative": 1 if doc and doc[0].pos_ == "VERB" else 0, "root_verb_is_" + root_verb: 1,
        "noun_count": sum(1 for t in doc if t.pos_ == "NOUN"),
    }
    return features

def train_intent_classifier(training_data: list):
    texts, labels = zip(*training_data)
    feature_dicts = [extract_linguistic_features(t) for t in texts]
    pipeline = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    pipeline.fit(feature_dicts, labels)
    return pipeline

# --- NEW: Audio Transcription Function ---
def transcribe_audio_file(audio_path: str) -> str:
    """Handles audio file conversion to WAV and transcription."""
    try:
        sound = AudioSegment.from_file(audio_path)
        wav_io = io.BytesIO()
        sound.export(wav_io, format="wav")
        wav_io.seek(0)
    except Exception as e:
        return f"ERROR: Could not process audio file. Ensure ffmpeg is installed. Details: {e}"

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        try:
            print("🎤 Transcribing audio...")
            transcript = recognizer.recognize_google(audio_data)
            return transcript
        except sr.UnknownValueError:
            return "ERROR: Google Speech Recognition could not understand the audio."
        except sr.RequestError as e:
            return f"ERROR: Could not request results from Google; {e}"

# --- Train the Classifier and Initialize the Searcher at Startup ---
print("--- [Setup Phase: Training Custom Intent Classifier] ---")
intent_classifier = train_intent_classifier(ANNOTATED_DATA)
print("✅ Custom classifier trained.")
print("\n--- [Setup Phase: Initializing Private Semantic Engine] ---")
semantic_searcher = SemanticSearcher(GOLDEN_ADVICE_LIBRARY)
print("✅ Semantic searcher initialized.")

# ==============================================================================
#  3. THE PERSONALIZATION ENGINE
# ==============================================================================
def apply_personalization(analysis_results: list, preferences: dict) -> str:
    """Applies user-defined preferences to the list of extracted advice."""
    # **MODIFIED** to return a more helpful message.
    if not analysis_results:
        return "No specific actionable advice was extracted from the voice note. For best results, please try using simpler, more direct language."

    focus_keywords = preferences.get('focus_keywords', '').lower().strip()
    focused_advice, other_advice = [], []

    if focus_keywords:
        focus_vector = nlp(focus_keywords).vector
        for advice in analysis_results:
            advice_vector = nlp(advice).vector
            if semantic_searcher._cosine_similarity(advice_vector, focus_vector) > 0.4:
                focused_advice.append(advice)
            else:
                other_advice.append(advice)
    else:
        other_advice = analysis_results

    summary_parts = []
    tone_prefix = "A good area to focus on is: " if preferences.get('tone') == 'encouraging' else ""

    if focused_advice:
        summary_parts.append("## 🎯 Feedback on Your Focus Areas")
        for advice in sorted(list(set(focused_advice))):
            summary_parts.append(f"- {tone_prefix}{advice}")
    
    if other_advice:
        summary_parts.append("\n## 📝 Other Key Takeaways")
        for advice in sorted(list(set(other_advice))):
            summary_parts.append(f"- {tone_prefix}{advice}")
            
    return "\n".join(summary_parts)

# ==============================================================================
#  4. FLASK ROUTES AND HTML TEMPLATES
# ==============================================================================

# --- HTML Template for the Home Page ---
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>compreheND - Neurodiversity Coaching Tool</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- OpenDyslexic font -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/open-dyslexic@0.0.1/open-dyslexic.min.css" />
    <style>
        body { font-family: 'Lexend', 'OpenDyslexic', 'Inter', Arial, Verdana, sans-serif; background-color: #f3f8fc; }
        .font-opendyslexic { font-family: 'OpenDyslexic', Arial, Verdana, sans-serif !important; }
        .font-lexend { font-family: 'Lexend', Arial, Verdana, sans-serif !important; }
        .font-sans { font-family: 'Inter', Arial, Verdana, sans-serif !important; }
        .settings-panel { background: #e6f2ff; border: 1px solid #b3d8ff; }
        .focus-outline:focus { outline: 3px solid #2563eb; outline-offset: 2px; }
        .dark-mode { background-color: #101624 !important; color: #e0e7ef !important; }
        .dark-mode .settings-panel { background: #1e293b !important; border-color: #334155 !important; }
        .dark-mode .bg-white { background-color: #1e293b !important; }
        .dark-mode .text-blue-800, .dark-mode .text-blue-700, .dark-mode .text-blue-900 { color: #e0e7ef !important; }
        .dark-mode .border-blue-200 { border-color: #334155 !important; }
        .dark-mode .bg-blue-50 { background-color: #101624 !important; }
        .dark-mode .bg-blue-100 { background-color: #334155 !important; }
        .dark-mode .bg-blue-200 { background-color: #334155 !important; }
        .dark-mode .bg-blue-700 { background-color: #2563eb !important; }
        .dark-mode .hover\:bg-blue-800:hover { background-color: #1e40af !important; }
        .dark-mode .file\:bg-blue-100 { background-color: #334155 !important; }
        .dark-mode .file\:text-blue-800 { color: #e0e7ef !important; }
        .dark-mode .border { border-color: #334155 !important; }
    </style>
    <!-- Lexend font -->
    <link href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;700&display=swap" rel="stylesheet">
</head>
<body class="bg-blue-50 text-blue-900 transition-all" id="mainBody">
    <div class="container mx-auto p-4 md:p-8 max-w-2xl">
        <div class="bg-white rounded-lg shadow-lg p-6 md:p-8 border-2 border-blue-200">
            <div class="flex items-center justify-between mb-4">
                <h1 class="text-3xl font-bold text-blue-800">compreheND</h1>
                <button id="settingsBtn" class="focus-outline bg-blue-100 hover:bg-blue-200 text-blue-700 px-3 py-1 rounded transition" aria-label="Open settings">⚙️ Settings</button>
            </div>
            <p class="text-blue-700 mb-6">A neurodiversity-friendly tool to translate a coach's voice note into clear, personalized advice.</p>

            <!-- Settings Panel -->
            <div id="settingsPanel" class="settings-panel rounded-lg p-4 mb-6 hidden" aria-label="Accessibility settings">
                <h2 class="text-lg font-semibold mb-2 text-blue-800">Accessibility & Appearance</h2>
                <div class="mb-3">
                    <label for="fontSelect" class="block text-blue-700 font-bold mb-1">Font Style</label>
                    <select id="fontSelect" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                        <option value="font-lexend">Lexend (Recommended)</option>
                        <option value="font-opendyslexic">OpenDyslexic</option>
                        <option value="font-sans">Sans-serif (Inter)</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="fontSizeSelect" class="block text-blue-700 font-bold mb-1">Font Size</label>
                    <select id="fontSizeSelect" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                        <option value="text-base">Medium</option>
                        <option value="text-lg">Large</option>
                        <option value="text-xl">Extra Large</option>
                    </select>
                </div>
                <div>
                    <label for="themeSelect" class="block text-blue-700 font-bold mb-1">Theme</label>
                    <select id="themeSelect" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                        <option value="system">System</option>
                        <option value="light">Light</option>
                        <option value="dark">Dark</option>
                    </select>
                </div>
            </div>

            <!-- Flash messages for errors -->
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
                    <strong class="font-bold">Error:</strong>
                    <span class="block sm:inline">{{ message }}</span>
                  </div>
                {% endfor %}
              {% endif %}
            {% endwith %}

            <form action="/analyze" method="post" enctype="multipart/form-data">
                <!-- Personalization Settings -->
                <div class="mb-6">
                    <h2 class="text-xl font-semibold mb-4 border-b pb-2 text-blue-800">1. Set Your Preferences</h2>
                    <div class="mb-4">
                        <label class="block text-blue-700 font-bold mb-2">Feedback Tone</label>
                        <select name="tone" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                            <option value="neutral">Neutral & Direct</option>
                            <option value="encouraging">Encouraging</option>
                        </select>
                    </div>
                    <div>
                        <label for="focus_keywords" class="block text-blue-700 font-bold mb-2">Focus Keywords (Optional)</label>
                        <input type="text" name="focus_keywords" id="focus_keywords" placeholder="e.g., defense, footwork" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                    </div>
                </div>

                <!-- Input Area for File Upload -->
                <div>
                    <h2 class="text-xl font-semibold mb-4 border-b pb-2 text-blue-800">2. Upload Coach's Voice Note</h2>
                    <input type="file" name="audio_file" accept="audio/*" required class="w-full text-sm text-blue-700 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-100 file:text-blue-800 hover:file:bg-blue-200 focus-outline"/>
                </div>

                <!-- Submit Button -->
                <div class="mt-6">
                    <button type="submit" class="w-full bg-blue-700 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-800 transition duration-300 focus-outline">
                        Analyze Voice Note
                    </button>
                </div>
            </form>
        </div>
    </div>
    <script>
        // Settings panel toggle
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsPanel = document.getElementById('settingsPanel');
        settingsBtn.addEventListener('click', () => {
            settingsPanel.classList.toggle('hidden');
        });
        // Font and size controls
        const fontSelect = document.getElementById('fontSelect');
        const fontSizeSelect = document.getElementById('fontSizeSelect');
        const themeSelect = document.getElementById('themeSelect');
        const mainBody = document.getElementById('mainBody');
        // Theme logic
        function applyTheme() {
            let theme = localStorage.getItem('nd_theme') || 'system';
            if (theme === 'system') {
                theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            }
            if (theme === 'dark') {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
            themeSelect.value = localStorage.getItem('nd_theme') || 'system';
        }
        themeSelect.addEventListener('change', () => {
            localStorage.setItem('nd_theme', themeSelect.value);
            applyTheme();
        });
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
        // Load preferences
        function applyPrefs() {
            const font = localStorage.getItem('nd_font') || 'font-lexend';
            const size = localStorage.getItem('nd_fontsize') || 'text-base';
            mainBody.classList.remove('font-opendyslexic', 'font-lexend', 'font-sans', 'text-base', 'text-lg', 'text-xl');
            mainBody.classList.add(font, size);
            fontSelect.value = font;
            fontSizeSelect.value = size;
            applyTheme();
        }
        fontSelect.addEventListener('change', () => {
            localStorage.setItem('nd_font', fontSelect.value);
            applyPrefs();
        });
        fontSizeSelect.addEventListener('change', () => {
            localStorage.setItem('nd_fontsize', fontSizeSelect.value);
            applyPrefs();
        });
        document.addEventListener('DOMContentLoaded', applyPrefs);
    </script>
</body>
</html>
"""

# --- HTML Template for the Results Page ---
RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>compreheND - Analysis Results</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/open-dyslexic@0.0.1/open-dyslexic.min.css" />
    <style>
        body { font-family: 'Lexend', 'OpenDyslexic', 'Inter', Arial, Verdana, sans-serif; background-color: #f3f8fc; }
        .font-opendyslexic { font-family: 'OpenDyslexic', Arial, Verdana, sans-serif !important; }
        .font-lexend { font-family: 'Lexend', Arial, Verdana, sans-serif !important; }
        .font-sans { font-family: 'Inter', Arial, Verdana, sans-serif !important; }
        .settings-panel { background: #e6f2ff; border: 1px solid #b3d8ff; }
        .focus-outline:focus { outline: 3px solid #2563eb; outline-offset: 2px; }
        .prose h2 { margin-top: 1.5em; margin-bottom: 0.5em; font-size: 1.25em; font-weight: 600; color: #2563eb; }
        .prose ul { list-style-type: none; padding-left: 0; }
        .prose li { background-color: #e0f2fe; border-left: 4px solid #2563eb; padding: 0.75em 1em; margin-bottom: 0.75em; border-radius: 0.25rem; color: #1e293b; }
        .dark-mode { background-color: #101624 !important; color: #e0e7ef !important; }
        .dark-mode .settings-panel { background: #1e293b !important; border-color: #334155 !important; }
        .dark-mode .bg-white { background-color: #1e293b !important; }
        .dark-mode .text-blue-800, .dark-mode .text-blue-700, .dark-mode .text-blue-900 { color: #e0e7ef !important; }
        .dark-mode .border-blue-200 { border-color: #334155 !important; }
        .dark-mode .bg-blue-50 { background-color: #101624 !important; }
        .dark-mode .bg-blue-100 { background-color: #334155 !important; }
        .dark-mode .bg-blue-200 { background-color: #334155 !important; }
        .dark-mode .bg-blue-700 { background-color: #2563eb !important; }
        .dark-mode .hover\:bg-blue-800:hover { background-color: #1e40af !important; }
        .dark-mode .file\:bg-blue-100 { background-color: #334155 !important; }
        .dark-mode .file\:text-blue-800 { color: #e0e7ef !important; }
        .dark-mode .border { border-color: #334155 !important; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;700&display=swap" rel="stylesheet">
</head>
<body class="bg-blue-50 text-blue-900 transition-all" id="mainBody">
    <div class="container mx-auto p-4 md:p-8 max-w-2xl">
        <div class="bg-white rounded-lg shadow-lg p-6 md:p-8 border-2 border-blue-200">
            <div class="flex items-center justify-between mb-4">
                <h1 class="text-3xl font-bold text-blue-800">compreheND</h1>
                <button id="settingsBtn" class="focus-outline bg-blue-100 hover:bg-blue-200 text-blue-700 px-3 py-1 rounded transition" aria-label="Open settings">⚙️ Settings</button>
            </div>
            <p class="text-blue-700 mb-6">Your personalized summary, based on the uploaded audio and your preferences.</p>
            <!-- Settings Panel -->
            <div id="settingsPanel" class="settings-panel rounded-lg p-4 mb-6 hidden" aria-label="Accessibility settings">
                <h2 class="text-lg font-semibold mb-2 text-blue-800">Accessibility & Appearance</h2>
                <div class="mb-3">
                    <label for="fontSelect" class="block text-blue-700 font-bold mb-1">Font Style</label>
                    <select id="fontSelect" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                        <option value="font-lexend">Lexend (Recommended)</option>
                        <option value="font-opendyslexic">OpenDyslexic</option>
                        <option value="font-sans">Sans-serif (Inter)</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="fontSizeSelect" class="block text-blue-700 font-bold mb-1">Font Size</label>
                    <select id="fontSizeSelect" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                        <option value="text-base">Medium</option>
                        <option value="text-lg">Large</option>
                        <option value="text-xl">Extra Large</option>
                    </select>
                </div>
                <div>
                    <label for="themeSelect" class="block text-blue-700 font-bold mb-1">Theme</label>
                    <select id="themeSelect" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                        <option value="system">System</option>
                        <option value="light">Light</option>
                        <option value="dark">Dark</option>
                    </select>
                </div>
            </div>
            <div class="prose max-w-none">
                {{ summary_html|safe }}
            </div>
            <div class="mt-8 border-t pt-6">
                <h3 class="text-lg font-semibold text-blue-700 mb-2">Original Transcript:</h3>
                <p class="text-sm text-blue-800 bg-blue-50 p-4 rounded-md border"><em>"{{ original_transcript }}"</em></p>
            </div>
            <div class="mt-8 text-center">
                <a href="/" class="bg-blue-700 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-800 transition duration-300 focus-outline">
                    Analyze New Voice Note
                </a>
            </div>
        </div>
    </div>
    <script>
        // Settings panel toggle
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsPanel = document.getElementById('settingsPanel');
        settingsBtn.addEventListener('click', () => {
            settingsPanel.classList.toggle('hidden');
        });
        // Font and size controls
        const fontSelect = document.getElementById('fontSelect');
        const fontSizeSelect = document.getElementById('fontSizeSelect');
        const themeSelect = document.getElementById('themeSelect');
        const mainBody = document.getElementById('mainBody');
        // Theme logic
        function applyTheme() {
            let theme = localStorage.getItem('nd_theme') || 'system';
            if (theme === 'system') {
                theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            }
            if (theme === 'dark') {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
            themeSelect.value = localStorage.getItem('nd_theme') || 'system';
        }
        themeSelect.addEventListener('change', () => {
            localStorage.setItem('nd_theme', themeSelect.value);
            applyTheme();
        });
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
        // Load preferences
        function applyPrefs() {
            const font = localStorage.getItem('nd_font') || 'font-lexend';
            const size = localStorage.getItem('nd_fontsize') || 'text-base';
            mainBody.classList.remove('font-opendyslexic', 'font-lexend', 'font-sans', 'text-base', 'text-lg', 'text-xl');
            mainBody.classList.add(font, size);
            fontSelect.value = font;
            fontSizeSelect.value = size;
            applyTheme();
        }
        fontSelect.addEventListener('change', () => {
            localStorage.setItem('nd_font', fontSelect.value);
            applyPrefs();
        });
        fontSizeSelect.addEventListener('change', () => {
            localStorage.setItem('nd_fontsize', fontSizeSelect.value);
            applyPrefs();
        });
        document.addEventListener('DOMContentLoaded', applyPrefs);
    </script>
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template_string(HOME_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    The main endpoint that handles the uploaded audio file.
    """
    # --- 1. Handle File Upload ---
    if 'audio_file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('home'))
    
    file = request.files['audio_file']
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # --- 2. Run Transcription ---
        transcript = transcribe_audio_file(filepath)
        
        # --- Clean up the temporary file ---
        os.remove(filepath)

        if "ERROR:" in transcript:
            flash(transcript)
            return redirect(url_for('home'))
        
        # --- 3. Get Preferences and Run NLP Analysis ---
        preferences = {
            'tone': request.form['tone'],
            'focus_keywords': request.form['focus_keywords']
        }
        
        doc = nlp(transcript)
        actionable_advice = []
        for sentence in doc.sents:
            features = extract_linguistic_features(sentence.text)
            predicted_intent = intent_classifier.predict([features])[0]
            if predicted_intent == "Actionable":
                concept = semantic_searcher.find_closest_advice(sentence.text)
                if "Novel advice" not in concept:
                    actionable_advice.append(concept)
        
        # --- 4. Apply Personalization ---
        summary_text = apply_personalization(actionable_advice, preferences)

        # --- 5. Render Results ---
        # **MODIFIED** to handle the "no advice" message gracefully.
        if summary_text.startswith("No specific actionable advice"):
            summary_html = f'<div class="text-center text-gray-500 italic py-4 bg-gray-50 rounded-lg border">{summary_text}</div>'
        else:
            # Convert markdown-style summary to basic HTML for display
            summary_html = ""
            lines = summary_text.split('\n')
            in_list = False
            for line in lines:
                if line.startswith('## '):
                    if in_list:
                        summary_html += '</ul>'
                        in_list = False
                    summary_html += f"<h2>{line[3:]}</h2>"
                elif line.startswith('- '):
                    if not in_list:
                        summary_html += '<ul>'
                        in_list = True
                    summary_html += f"<li>{line[2:]}</li>"
            if in_list:
                summary_html += '</ul>'

        return render_template_string(RESULTS_TEMPLATE, summary_html=summary_html, original_transcript=transcript)

    return redirect(url_for('home'))

# --- Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True)

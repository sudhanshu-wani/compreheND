# --- Core Imports ---
from flask import Flask, render_template_string, request, flash, redirect, url_for
import spacy
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from textblob import TextBlob
import os
import io
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import joblib

# --- Advanced Component Imports ---
from faster_whisper import WhisperModel
import librosa
import noisereduce as nr
import soundfile as sf
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# ==============================================================================
#  1. INITIALIZE FLASK APP AND LOAD MODELS
# ==============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_strong_secret_key_for_msc_project'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CLASSIFIER_PATH = 'intent_classifier.pkl'

# --- Load spaCy Model ---
print("📚 Loading spaCy medium model...")
nlp = spacy.load("en_core_web_md")
print("✅ spaCy model loaded.")

# --- Load Whisper Model ---
# Using the tiny, English-only model for efficiency on a low-end PC.
# It will be downloaded automatically on the first run.
model_size = "tiny.en"
print(f"📚 Loading faster-whisper model: {model_size}...")
whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("✅ Whisper model loaded.")

# ==============================================================================
#  2. THE CORE AI PIPELINE (Upgraded Components)
# ==============================================================================

# --- The Enhanced Golden Advice Library ---
GOLDEN_ADVICE_LIBRARY = [
    # Technical Skills
    {"advice": "player is unfocused and not following the game plan", "concept": "Improve tactical discipline and adhere to the team strategy."},
    {"advice": "player is holding onto the ball or possession for too long", "concept": "Increase speed of play; focus on quicker decisions and team integration."},
    {"advice": "player is making poor decisions and forcing plays that are not there", "concept": "Improve situational awareness and shot/pass selection."},
    {"advice": "player's first touch is too heavy and losing possession", "concept": "Focus on softer ball control and cushioning the first touch."},
    {"advice": "player is not maintaining proper defensive stance", "concept": "Keep low center of gravity and stay on balls of feet for better defensive mobility."},
    
    # Mental Game
    {"advice": "player seems anxious and rushing decisions under pressure", "concept": "Practice breathing techniques and pre-performance routines to stay calm."},
    {"advice": "player loses focus after making mistakes", "concept": "Develop resilience through positive self-talk and next-play mentality."},
    {"advice": "player is hesitant to take initiative in key moments", "concept": "Build confidence through gradual exposure to pressure situations in practice."},
    
    # Team Dynamics
    {"advice": "player is not communicating effectively with teammates", "concept": "Increase verbal communication and use clear, specific callouts."},
    {"advice": "player is not supporting teammates in transition", "concept": "Improve off-ball movement and anticipate teammates' needs."},
    {"advice": "player shows frustration with teammates' mistakes", "concept": "Practice empathy and constructive communication with teammates."},
    
    # Game Strategy
    {"advice": "player is not recognizing defensive patterns", "concept": "Study game film and practice pattern recognition in training."},
    {"advice": "player is not adapting to opponent's strategy", "concept": "Develop flexibility in tactical approach and read game situations better."},
    {"advice": "player is not managing energy effectively", "concept": "Work on pacing and strategic rest during natural game breaks."},
    
    # Leadership
    {"advice": "player is not taking responsibility in crucial moments", "concept": "Embrace leadership opportunities and trust in your abilities."},
    {"advice": "player is not helping organize the team", "concept": "Take initiative in team organization and tactical adjustments."},
    {"advice": "player shows negative body language", "concept": "Maintain positive body language to boost team morale."}
]

# --- Enhanced Annotated Dataset for the Intent Classifier ---
ANNOTATED_DATA = [
    # Technical Instructions (Actionable)
    ("Just play the simple pass.", "Actionable"),
    ("Keep your shoulders square to the target.", "Actionable"),
    ("Move your feet faster to get into position.", "Actionable"),
    ("Watch the ball all the way onto your bat.", "Actionable"),
    ("Stay low when defending.", "Actionable"),
    ("Follow through with your swing.", "Actionable"),
    ("Keep your head still while making contact.", "Actionable"),
    
    # Mental Game Instructions (Actionable)
    ("Take a deep breath before serving.", "Actionable"),
    ("Focus on one point at a time.", "Actionable"),
    ("Trust your training and stick to the basics.", "Actionable"),
    ("Visualize your successful shots before the match.", "Actionable"),
    
    # Strategic Instructions (Actionable)
    ("Look for gaps in their defense.", "Actionable"),
    ("Change up your serve placement.", "Actionable"),
    ("Make them play to your strengths.", "Actionable"),
    ("Force them to their weaker side.", "Actionable"),
    
    # Team Communication (Actionable)
    ("Call for the ball early and clearly.", "Actionable"),
    ("Signal your intentions to your teammates.", "Actionable"),
    ("Direct traffic on defense.", "Actionable"),
    
    # Constructive Criticism (Actionable)
    ("Your footwork needs to be quicker.", "Actionable"),
    ("You're dropping your elbow too early.", "Actionable"),
    ("Your grip is too tight on the racket.", "Actionable"),
    
    # Praise and Observations (Non-Actionable)
    ("That was a fantastic effort.", "Non-Actionable"),
    ("You're really improving each week.", "Non-Actionable"),
    ("I'm impressed with your dedication.", "Non-Actionable"),
    ("Your energy today is outstanding.", "Non-Actionable"),
    ("You've got natural talent.", "Non-Actionable"),
    
    # Sarcasm/Non-Constructive (Non-Actionable)
    ("Well, that was interesting.", "Non-Actionable"),
    ("Your teammates might as well be selling popcorn in the stands.", "Non-Actionable"),
    ("Honestly, your determination to defend every single ball is admirable, truly.", "Non-Actionable"),
    ("I've seen better coordination in a nursery.", "Non-Actionable"),
    
    # General Comments (Non-Actionable)
    ("The weather is affecting everyone's game today.", "Non-Actionable"),
    ("These are tough conditions to play in.", "Non-Actionable"),
    ("The opposition is very experienced.", "Non-Actionable"),
    ("We've got a challenging schedule ahead.", "Non-Actionable")
]

# --- Semantic Searcher Class (Unchanged) ---
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

# --- UPGRADED Intent Classifier Functions ---
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
    """Trains a more powerful LightGBM classifier and saves it to a file."""
    texts, labels = zip(*training_data)
    feature_dicts = [extract_linguistic_features(t) for t in texts]
    pipeline = make_pipeline(DictVectorizer(), lgb.LGBMClassifier(objective='binary', class_weight='balanced'))
    pipeline.fit(feature_dicts, labels)
    # Save the trained model for future use
    joblib.dump(pipeline, CLASSIFIER_PATH)
    return pipeline

# --- UPGRADED Audio Transcription with Pre-processing ---
def transcribe_audio_file(audio_path: str) -> str:
    """Handles audio pre-processing and local transcription with Whisper."""
    try:
        print(f"Processing audio file: {audio_path}")
        
        # 1. First try to load with librosa (handles most formats)
        try:
            audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            print(f"Successfully loaded audio with librosa. Sample rate: {sample_rate}")
        except Exception as librosa_error:
            print(f"Librosa failed: {librosa_error}")
            # 2. Fallback: Use pydub to convert to WAV first
            try:
                from pydub import AudioSegment
                print("Attempting conversion with pydub...")
                audio_segment = AudioSegment.from_file(audio_path)
                # Export as WAV to a temporary file
                temp_wav_path = audio_path + "_temp.wav"
                audio_segment.export(temp_wav_path, format="wav")
                # Load the converted WAV file
                audio, sample_rate = librosa.load(temp_wav_path, sr=16000, mono=True)
                # Clean up temp file
                import os
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                print("Successfully converted and loaded audio with pydub")
            except Exception as pydub_error:
                return f"ERROR: Failed to load audio file. Both librosa and pydub failed.\nLibrosa error: {librosa_error}\nPydub error: {pydub_error}"
        
        # 3. Perform noise reduction
        print("Reducing noise from audio...")
        try:
            reduced_noise_audio = nr.reduce_noise(y=audio, sr=sample_rate)
        except Exception as noise_error:
            print(f"Noise reduction failed: {noise_error}")
            # Continue without noise reduction
            reduced_noise_audio = audio
        
        # 4. Transcribe the cleaned audio using faster-whisper
        print("🎤 Transcribing audio with faster-whisper...")
        segments, _ = whisper_model.transcribe(reduced_noise_audio, beam_size=5)
        
        transcript = " ".join([segment.text for segment in segments])
        return transcript.strip()

    except Exception as e:
        return f"ERROR: Failed during audio processing or transcription. Details: {e}"

# --- Train or Load Classifier and Initialize Searcher at Startup ---
if os.path.exists(CLASSIFIER_PATH):
    print(f"--- [Setup Phase: Loading existing classifier from {CLASSIFIER_PATH}] ---")
    intent_classifier = joblib.load(CLASSIFIER_PATH)
    print("✅ Custom classifier loaded.")
else:
    print("--- [Setup Phase: Training new classifier (first run)] ---")
    intent_classifier = train_intent_classifier(ANNOTATED_DATA)
    print("✅ Custom classifier trained and saved.")

print("\n--- [Setup Phase: Initializing Private Semantic Engine] ---")
semantic_searcher = SemanticSearcher(GOLDEN_ADVICE_LIBRARY)
print("✅ Semantic searcher initialized.")

# ==============================================================================
#  3. THE PERSONALIZATION ENGINE (Unchanged)
# ==============================================================================
def apply_personalization(analysis_results: list, preferences: dict) -> str:
    if not analysis_results:
        return "No specific actionable advice was extracted. For best results, please try using simpler, more direct language in a quieter environment."
    
    focus_keywords = preferences.get('focus_keywords', '').lower().strip()
    response_length = preferences.get('response_length', 'medium')
    audio_cues = preferences.get('audio_cues', 'no')
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
    
    # Apply response length formatting
    def format_advice(advice_list, section_title):
        if not advice_list:
            return []
        
        formatted_section = [section_title]
        unique_advice = sorted(list(set(advice_list)))
        
        for advice in unique_advice:
            if response_length == 'short':
                # Extract key action and important nouns from the advice
                doc = nlp(advice)
                
                # Find the main verb
                main_verb = ""
                for token in doc:
                    if token.pos_ in ['VERB', 'AUX'] and token.dep_ in ['ROOT', 'aux']:
                        main_verb = token.text
                        break
                
                # Find important nouns (subjects and objects)
                important_nouns = []
                for token in doc:
                    if (token.pos_ == 'NOUN' and 
                        token.dep_ in ['nsubj', 'dobj', 'pobj', 'compound'] and
                        len(token.text) > 2):  # Avoid very short words
                        important_nouns.append(token.text)
                
                # Create concise bullet point with verb + key nouns
                if main_verb and important_nouns:
                    # Take up to 2 most important nouns
                    key_terms = important_nouns[:2]
                    formatted_advice = f"- {tone_prefix}{main_verb.capitalize()} {', '.join(key_terms)}"
                elif main_verb:
                    # If no nouns found, use verb + first few meaningful words
                    words = [word for word in advice.split() if len(word) > 3][:2]
                    if words:
                        formatted_advice = f"- {tone_prefix}{main_verb.capitalize()} {words[0]}"
                    else:
                        formatted_advice = f"- {tone_prefix}{main_verb.capitalize()}"
                else:
                    # Fallback: use first 3-4 meaningful words
                    words = [word for word in advice.split() if len(word) > 2][:4]
                    formatted_advice = f"- {tone_prefix}{' '.join(words).capitalize()}"
                    
            else:  # medium length
                # Use the full advice with tone prefix
                formatted_advice = f"- {tone_prefix}{advice}"
            
            formatted_section.append(formatted_advice)
        
        return formatted_section
    
    # Format focused advice
    if focused_advice:
        summary_parts.extend(format_advice(focused_advice, "## 🎯 Feedback on Your Focus Areas"))
    
    # Format other advice
    if other_advice:
        summary_parts.extend(format_advice(other_advice, "\n## 📝 Other Key Takeaways"))
    
    return "\n".join(summary_parts)

# ==============================================================================
#  4. FLASK ROUTES AND HTML TEMPLATES (Unchanged)
# ==============================================================================
# (The HTML templates and Flask routes are identical to the previous version)

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
                    <div class="mb-4">
                        <label class="block text-blue-700 font-bold mb-2">Response Length</label>
                        <select name="response_length" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                            <option value="short">Short (Concise bullet points)</option>
                            <option value="medium">Medium (Detailed explanations)</option>
                        </select>
                    </div>
                    <div class="mb-4">
                        <label class="block text-blue-700 font-bold mb-2">Audio Cues</label>
                        <select name="audio_cues" class="w-full p-2 border rounded-md bg-blue-50 focus-outline">
                            <option value="no">No audio cues</option>
                            <option value="yes">Yes, add speaker buttons for each advice</option>
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
                    <button type="submit" id="submitBtn" class="w-full bg-blue-700 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-800 transition duration-300 focus-outline disabled:opacity-50 disabled:cursor-not-allowed">
                        <span id="buttonText" class="inline-flex items-center">Analyze Voice Note</span>
                        <span id="loadingSpinner" class="hidden inline-flex items-center">
                            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Processing...
                        </span>
                    </button>
                </div>
                <!-- Privacy Policy and Terms of Use -->
                <div class="mt-8 pt-6 border-t border-blue-200">
                    <p class="text-sm text-blue-700 text-center">
                        By using this tool, you agree to our 
                        <button id="privacyBtn" class="text-blue-800 underline hover:text-blue-900 focus-outline" type="button">Privacy Policy</button> 
                        and 
                        <button id="termsBtn" class="text-blue-800 underline hover:text-blue-900 focus-outline" type="button">Terms of Use</button>.
                    </p>
                </div>
            </form>
        </div>
    </div>

    <!-- Privacy Policy Modal -->
    <div id="privacyModal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50 flex items-center justify-center p-4">
        <div class="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div class="p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold text-blue-800">Privacy Policy</h2>
                    <button id="closePrivacy" class="text-gray-500 hover:text-gray-700 text-2xl focus-outline">&times;</button>
                </div>
                <div class="text-sm">
                    <h3 class="text-lg font-semibold text-blue-700 mb-2">Data Protection & Privacy</h3>
                    <p class="mb-3">This application is designed with your privacy in mind and complies with GDPR and UK data protection regulations.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Data We Process:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>Audio files you upload for analysis</li>
                        <li>Transcribed text from your audio</li>
                        <li>Your preferences (tone, focus keywords)</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">How We Use Your Data:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>To transcribe your audio using local processing</li>
                        <li>To analyze and provide personalized coaching feedback</li>
                        <li>To improve our service functionality</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Data Storage & Security:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>Audio files are temporarily stored during processing and automatically deleted</li>
                        <li>No personal data is permanently stored on our servers</li>
                        <li>All processing is done locally on your device where possible</li>
                        <li>We use industry-standard security measures to protect your data</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Your Rights (GDPR):</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>Right to access your personal data</li>
                        <li>Right to rectification of inaccurate data</li>
                        <li>Right to erasure ("right to be forgotten")</li>
                        <li>Right to data portability</li>
                        <li>Right to object to processing</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Contact Information:</h4>
                    <p class="mb-3">For privacy-related inquiries, please contact us at: privacy@comprehend-tool.com</p>
                    
                    <p class="text-xs text-gray-600 mt-4">Last updated: January 2025</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Terms of Use Modal -->
    <div id="termsModal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50 flex items-center justify-center p-4">
        <div class="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div class="p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold text-blue-800">Terms of Use</h2>
                    <button id="closeTerms" class="text-gray-500 hover:text-gray-700 text-2xl focus-outline">&times;</button>
                </div>
                <div class="text-sm">
                    <h3 class="text-lg font-semibold text-blue-700 mb-2">Acceptance of Terms</h3>
                    <p class="mb-3">By using this neurodiversity coaching tool, you agree to be bound by these Terms of Use.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Service Description:</h4>
                    <p class="mb-3">This tool provides AI-powered analysis of coaching voice notes to generate personalized feedback for neurodiverse individuals.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">User Responsibilities:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>You must own or have permission to use any audio content you upload</li>
                        <li>You are responsible for the accuracy and appropriateness of your content</li>
                        <li>You must not upload malicious files or attempt to compromise the service</li>
                        <li>You must respect intellectual property rights</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Limitations of Service:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>This tool is for educational and coaching purposes only</li>
                        <li>Results are AI-generated and should not replace professional advice</li>
                        <li>We do not guarantee accuracy of transcriptions or analysis</li>
                        <li>Service availability may vary and is provided "as is"</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Intellectual Property:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>You retain ownership of your uploaded content</li>
                        <li>Analysis results are provided for your personal use</li>
                        <li>Our AI models and algorithms remain our intellectual property</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Liability & Disclaimers:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>We are not liable for any decisions made based on our analysis</li>
                        <li>We do not guarantee uninterrupted service</li>
                        <li>We are not responsible for any indirect or consequential damages</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Changes to Terms:</h4>
                    <p class="mb-3">We reserve the right to modify these terms at any time. Continued use constitutes acceptance of updated terms.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Governing Law:</h4>
                    <p class="mb-3">These terms are governed by UK law and subject to UK jurisdiction.</p>
                    
                    <p class="text-xs text-gray-600 mt-4">Last updated: January 2025</p>
                </div>
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
        // Load preferences when page loads
        document.addEventListener('DOMContentLoaded', function() {
            applyPrefs();
            loadAudioPreferences();
        });
        
        // Form submission handling
        const form = document.querySelector('form');
        const submitBtn = document.getElementById('submitBtn');
        const buttonText = document.getElementById('buttonText');
        const loadingSpinner = document.getElementById('loadingSpinner');

        form.addEventListener('submit', (e) => {
            // Only proceed if a file is selected
            const fileInput = document.querySelector('input[type="file"]');
            if (fileInput.files.length > 0) {
                // Disable the button and show loading state
                submitBtn.disabled = true;
                buttonText.classList.add('hidden');
                loadingSpinner.classList.remove('hidden');
            }
        });
        
        // Privacy Policy and Terms of Use Modal Functionality
        const privacyBtn = document.getElementById('privacyBtn');
        const termsBtn = document.getElementById('termsBtn');
        const privacyModal = document.getElementById('privacyModal');
        const termsModal = document.getElementById('termsModal');
        const closePrivacy = document.getElementById('closePrivacy');
        const closeTerms = document.getElementById('closeTerms');
        
        // Open Privacy Policy Modal
        privacyBtn.addEventListener('click', () => {
            privacyModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        });
        
        // Open Terms of Use Modal
        termsBtn.addEventListener('click', () => {
            termsModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        });
        
        // Close Privacy Policy Modal
        closePrivacy.addEventListener('click', () => {
            privacyModal.classList.add('hidden');
            document.body.style.overflow = 'auto';
        });
        
        // Close Terms of Use Modal
        closeTerms.addEventListener('click', () => {
            termsModal.classList.add('hidden');
            document.body.style.overflow = 'auto';
        });
        
        // Close modals when clicking outside
        privacyModal.addEventListener('click', (e) => {
            if (e.target === privacyModal) {
                privacyModal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        });
        
        termsModal.addEventListener('click', (e) => {
            if (e.target === termsModal) {
                termsModal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        });
        
        // Close modals with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                privacyModal.classList.add('hidden');
                termsModal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        });
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
        /* Speaker button styling */
        .speaker-btn { transition: all 0.2s ease-in-out; }
        .speaker-btn:hover { transform: scale(1.1); }
        .speaker-btn:active { transform: scale(0.95); }
        .speaker-btn.speaking { animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        /* List item styling for audio cues */
        .audio-cue-item { background-color: #e0f2fe; border-left: 4px solid #2563eb; padding: 0.75em 1em; margin-bottom: 0.75em; border-radius: 0.25rem; color: #1e293b; }
        /* Range input styling */
        input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            background: transparent;
            cursor: pointer;
        }
        input[type="range"]::-webkit-slider-track {
            background: #dbeafe;
            height: 8px;
            border-radius: 4px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            background: #2563eb;
            height: 20px;
            width: 20px;
            border-radius: 50%;
            cursor: pointer;
        }
        input[type="range"]::-moz-range-track {
            background: #dbeafe;
            height: 8px;
            border-radius: 4px;
            border: none;
        }
        input[type="range"]::-moz-range-thumb {
            background: #2563eb;
            height: 20px;
            width: 20px;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }
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
                <div class="flex items-center space-x-2">
                    {% if audio_cues_enabled %}
                    <button id="stopAudioBtn" onclick="stopAllAudio()" class="focus-outline bg-red-100 hover:bg-red-200 text-red-700 px-3 py-1 rounded transition" title="Stop all audio playback">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M6 6h12v12H6z"/>
                        </svg>
                        Stop Audio
                    </button>
                    {% endif %}
                <button id="settingsBtn" class="focus-outline bg-blue-100 hover:bg-blue-200 text-blue-700 px-3 py-1 rounded transition" aria-label="Open settings">⚙️ Settings</button>
                </div>
            </div>
            <p class="text-blue-700 mb-6">Your personalized summary, based on the uploaded audio and your preferences.</p>
            {% if audio_cues_enabled %}
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <div class="flex items-center">
                    <svg class="w-5 h-5 text-blue-600 mr-2" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                    </svg>
                    <p class="text-blue-700 text-sm"><strong>Audio Cues Enabled:</strong> Click the speaker button next to each advice item to hear it spoken aloud.</p>
                </div>
            </div>
            {% endif %}
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
                {% if audio_cues_enabled %}
                <div class="border-t pt-4 mt-4">
                    <h3 class="text-md font-semibold mb-3 text-blue-800">Audio Settings</h3>
                    <div class="mb-3">
                        <label for="volumeControl" class="block text-blue-700 font-bold mb-1">Speech Volume</label>
                        <input type="range" id="volumeControl" min="0" max="1" step="0.1" value="1" 
                               class="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500"
                               oninput="setSpeechVolume(this.value)">
                        <div class="flex justify-between text-xs text-blue-600">
                            <span>Mute</span>
                            <span>Full</span>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label for="rateControl" class="block text-blue-700 font-bold mb-1">Speech Speed</label>
                        <input type="range" id="rateControl" min="0.5" max="2" step="0.1" value="0.9" 
                               class="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500"
                               oninput="setSpeechRate(this.value)">
                        <div class="flex justify-between text-xs text-blue-600">
                            <span>Slow</span>
                            <span>Fast</span>
                        </div>
                    </div>
                </div>
                {% endif %}
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
            
            <!-- Privacy Policy and Terms of Use -->
            <div class="mt-8 pt-6 border-t border-blue-200">
                <p class="text-sm text-blue-700 text-center">
                    By using this tool, you agree to our 
                    <button id="privacyBtn" class="text-blue-800 underline hover:text-blue-900 focus-outline" type="button">Privacy Policy</button> 
                    and 
                    <button id="termsBtn" class="text-blue-800 underline hover:text-blue-900 focus-outline" type="button">Terms of Use</button>.
                </p>
            </div>
        </div>
    </div>

    <!-- Privacy Policy Modal -->
    <div id="privacyModal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50 flex items-center justify-center p-4">
        <div class="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div class="p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold text-blue-800">Privacy Policy</h2>
                    <button id="closePrivacy" class="text-gray-500 hover:text-gray-700 text-2xl focus-outline">&times;</button>
                </div>
                <div class="text-sm">
                    <h3 class="text-lg font-semibold text-blue-700 mb-2">Data Protection & Privacy</h3>
                    <p class="mb-3">This application is designed with your privacy in mind and complies with GDPR and UK data protection regulations.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Data We Process:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>Audio files you upload for analysis</li>
                        <li>Transcribed text from your audio</li>
                        <li>Your preferences (tone, focus keywords)</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">How We Use Your Data:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>To transcribe your audio using local processing</li>
                        <li>To analyze and provide personalized coaching feedback</li>
                        <li>To improve our service functionality</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Data Storage & Security:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>Audio files are temporarily stored during processing and automatically deleted</li>
                        <li>No personal data is permanently stored on our servers</li>
                        <li>All processing is done locally on your device where possible</li>
                        <li>We use industry-standard security measures to protect your data</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Your Rights (GDPR):</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>Right to access your personal data</li>
                        <li>Right to rectification of inaccurate data</li>
                        <li>Right to erasure ("right to be forgotten")</li>
                        <li>Right to data portability</li>
                        <li>Right to object to processing</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Contact Information:</h4>
                    <p class="mb-3">For privacy-related inquiries, please contact us at: privacy@comprehend-tool.com</p>
                    
                    <p class="text-xs text-gray-600 mt-4">Last updated: January 2025</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Terms of Use Modal -->
    <div id="termsModal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50 flex items-center justify-center p-4">
        <div class="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div class="p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold text-blue-800">Terms of Use</h2>
                    <button id="closeTerms" class="text-gray-500 hover:text-gray-700 text-2xl focus-outline">&times;</button>
                </div>
                <div class="text-sm">
                    <h3 class="text-lg font-semibold text-blue-700 mb-2">Acceptance of Terms</h3>
                    <p class="mb-3">By using this neurodiversity coaching tool, you agree to be bound by these Terms of Use.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Service Description:</h4>
                    <p class="mb-3">This tool provides AI-powered analysis of coaching voice notes to generate personalized feedback for neurodiverse individuals.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">User Responsibilities:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>You must own or have permission to use any audio content you upload</li>
                        <li>You are responsible for the accuracy and appropriateness of your content</li>
                        <li>You must not upload malicious files or attempt to compromise the service</li>
                        <li>You must respect intellectual property rights</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Limitations of Service:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>This tool is for educational and coaching purposes only</li>
                        <li>Results are AI-generated and should not replace professional advice</li>
                        <li>We do not guarantee accuracy of transcriptions or analysis</li>
                        <li>Service availability may vary and is provided "as is"</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Intellectual Property:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>You retain ownership of your uploaded content</li>
                        <li>Analysis results are provided for your personal use</li>
                        <li>Our AI models and algorithms remain our intellectual property</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Liability & Disclaimers:</h4>
                    <ul class="list-disc pl-5 mb-3">
                        <li>We are not liable for any decisions made based on our analysis</li>
                        <li>We do not guarantee uninterrupted service</li>
                        <li>We are not responsible for any indirect or consequential damages</li>
                    </ul>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Changes to Terms:</h4>
                    <p class="mb-3">We reserve the right to modify these terms at any time. Continued use constitutes acceptance of updated terms.</p>
                    
                    <h4 class="font-semibold text-blue-600 mb-2">Governing Law:</h4>
                    <p class="mb-3">These terms are governed by UK law and subject to UK jurisdiction.</p>
                    
                    <p class="text-xs text-gray-600 mt-4">Last updated: January 2025</p>
                </div>
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
        // Load preferences when page loads
        document.addEventListener('DOMContentLoaded', function() {
            applyPrefs();
            loadAudioPreferences();
        });

        // Text-to-Speech functionality
        let currentSpeech = null;
        let speechVolume = 1.0;
        let speechRate = 0.9;
        
        function speakText(buttonEl, text) {
            // Stop any currently playing speech
            if (window.speechSynthesis && speechSynthesis.speaking) {
                speechSynthesis.cancel();
            }
            
            // Check if speech synthesis is supported
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = speechRate;
                utterance.pitch = 1.0;
                utterance.volume = speechVolume;
                
                // Ensure voices are loaded before speaking
                const ensureVoices = () => new Promise(resolve => {
                    let voices = speechSynthesis.getVoices();
                    if (voices && voices.length) return resolve(voices);
                    const iv = setInterval(() => {
                        voices = speechSynthesis.getVoices();
                        if (voices && voices.length) {
                            clearInterval(iv);
                            resolve(voices);
                        }
                    }, 50);
                });
                
                ensureVoices().then(voices => {
                    const englishVoice = voices.find(voice => 
                        voice.lang && voice.lang.startsWith('en') && /Google|Microsoft|Apple|English/i.test(voice.name)
                    ) || voices.find(voice => voice.lang && voice.lang.startsWith('en')) || voices[0];
                    if (englishVoice) utterance.voice = englishVoice;
                    
                    utterance.onstart = function() {
                        currentSpeech = utterance;
                        if (buttonEl) buttonEl.classList.add('text-green-600', 'bg-green-100', 'speaking');
                        if (buttonEl) showTooltip(buttonEl, 'Playing audio...');
                    };
                    
                    utterance.onend = function() {
                        currentSpeech = null;
                        if (buttonEl) buttonEl.classList.remove('text-green-600', 'bg-green-100', 'speaking');
                        if (buttonEl) showTooltip(buttonEl, 'Audio finished');
                    };
                    
                    utterance.onerror = function(e) {
                        currentSpeech = null;
                        if (buttonEl) buttonEl.classList.remove('text-green-600', 'bg-green-100', 'speaking');
                        console.error('Speech synthesis error:', e.error);
                        if (buttonEl) showTooltip(buttonEl, 'Audio Stopped');
                    };
                    
                    speechSynthesis.speak(utterance);
                });
            } else {
                alert('Text-to-speech is not supported in your browser. Please try a modern browser like Chrome, Firefox, or Edge.');
            }
        }
        
        // Function to adjust speech volume
        function setSpeechVolume(volume) {
            speechVolume = Math.max(0, Math.min(1, volume));
            localStorage.setItem('nd_speech_volume', speechVolume);
            console.log('Speech volume set to:', speechVolume);
        }
        
        // Function to adjust speech rate
        function setSpeechRate(rate) {
            speechRate = Math.max(0.5, Math.min(2, rate));
            localStorage.setItem('nd_speech_rate', speechRate);
            console.log('Speech rate set to:', speechRate);
        }
        
        // Function to load audio preferences
        function loadAudioPreferences() {
            const savedVolume = localStorage.getItem('nd_speech_volume');
            const savedRate = localStorage.getItem('nd_speech_rate');
            
            if (savedVolume !== null) {
                speechVolume = parseFloat(savedVolume);
                const volumeControl = document.getElementById('volumeControl');
                if (volumeControl) volumeControl.value = speechVolume;
            }
            
            if (savedRate !== null) {
                speechRate = parseFloat(savedRate);
                const rateControl = document.getElementById('rateControl');
                if (rateControl) rateControl.value = speechRate;
            }
        }
        
        // Load preferences when page loads
        document.addEventListener('DOMContentLoaded', function() {
            applyPrefs();
            loadAudioPreferences();
        });

        // Stop speech when navigating away or closing
        window.addEventListener('beforeunload', function() {
            if (window.speechSynthesis && speechSynthesis.speaking) {
                speechSynthesis.cancel();
            }
        });

        // Function to stop all speech synthesis
        function stopAllAudio() {
            if ('speechSynthesis' in window) {
                speechSynthesis.cancel();
                currentSpeech = null;
                // Remove speaking class from all buttons
                document.querySelectorAll('.speaking').forEach(btn => {
                    btn.classList.remove('text-green-600', 'bg-green-100', 'speaking');
                });
                console.log('All audio playback stopped.');
            } else {
                alert('Text-to-speech is not supported in your browser. Cannot stop audio.');
            }
        }
        
        // Function to show tooltip
        function showTooltip(element, message) {
            const tooltip = document.createElement('div');
            tooltip.className = 'fixed bg-gray-800 text-white text-sm px-2 py-1 rounded z-50 pointer-events-none';
            tooltip.textContent = message;
            tooltip.style.left = element.getBoundingClientRect().left + 'px';
            tooltip.style.top = (element.getBoundingClientRect().top - 30) + 'px';
            document.body.appendChild(tooltip);
            
            setTimeout(() => {
                if (tooltip.parentNode) {
                    tooltip.parentNode.removeChild(tooltip);
                }
            }, 2000);
        }

        // Privacy Policy and Terms of Use Modal Functionality
        const privacyBtn = document.getElementById('privacyBtn');
        const termsBtn = document.getElementById('termsBtn');
        const privacyModal = document.getElementById('privacyModal');
        const termsModal = document.getElementById('termsModal');
        const closePrivacy = document.getElementById('closePrivacy');
        const closeTerms = document.getElementById('closeTerms');
        
        // Open Privacy Policy Modal
        privacyBtn.addEventListener('click', () => {
            privacyModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        });
        
        // Open Terms of Use Modal
        termsBtn.addEventListener('click', () => {
            termsModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        });
        
        // Close Privacy Policy Modal
        closePrivacy.addEventListener('click', () => {
            privacyModal.classList.add('hidden');
            document.body.style.overflow = 'auto';
        });
        
        // Close Terms of Use Modal
        closeTerms.addEventListener('click', () => {
            termsModal.classList.add('hidden');
            document.body.style.overflow = 'auto';
        });
        
        // Close modals when clicking outside
        privacyModal.addEventListener('click', (e) => {
            if (e.target === privacyModal) {
                privacyModal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        });
        
        termsModal.addEventListener('click', (e) => {
            if (e.target === termsModal) {
                termsModal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        });
        
        // Close modals with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                privacyModal.classList.add('hidden');
                termsModal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    
    return render_template_string(HOME_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    
    if 'audio_file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('home'))
    
    file = request.files['audio_file']
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('home'))
    
    # Check file extension
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.au', '.wma', '.aiff'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        flash(f'Unsupported file format. Please use: {", ".join(allowed_extensions)}')
        return redirect(url_for('home'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Check if file was actually saved and has content
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                flash('Error: File upload failed or file is empty.')
                return redirect(url_for('home'))
            
            transcript = transcribe_audio_file(filepath)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('home'))
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

        if "ERROR:" in transcript:
            flash(transcript)
            return redirect(url_for('home'))
        
        preferences = {
            'tone': request.form['tone'], 
            'focus_keywords': request.form['focus_keywords'],
            'response_length': request.form['response_length'],
            'audio_cues': request.form['audio_cues']
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
        
        
        summary_text = apply_personalization(actionable_advice, preferences)

        
        if summary_text.startswith("No specific actionable advice"):
            summary_html = f'<div class="text-center text-gray-500 italic py-4 bg-gray-50 rounded-lg border">{summary_text}</div>'
        else:
      
            summary_html = ""
            lines = summary_text.split('\n')
            in_list = False
            for line in lines:
                if line.startswith('## '):
                    if in_list: summary_html += '</ul>'
                    in_list = False
                    summary_html += f"<h2>{line[3:]}</h2>"
                elif line.startswith('- '):
                    if not in_list:
                        summary_html += '<ul>'
                        in_list = True
                    
                    # Extract the advice text
                    advice_text = line[2:]
                     
                     # Add speaker button if audio cues are enabled
                    if preferences.get('audio_cues') == 'yes':
                        summary_html += f'<li class="audio-cue-item flex items-center justify-between group"><span class="flex-1">{advice_text}</span><button onclick="speakText(this, \'{advice_text.replace(chr(39), "&apos;")}\')" class="speaker-btn ml-3 p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-100 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500" title="Click to hear this advice" aria-label="Listen to: {advice_text.replace(chr(39), "&apos;")}" tabindex="0" onkeydown="if(event.key===\'Enter\'||event.key===\' \')speakText(this, \'{advice_text.replace(chr(39), "&apos;")}\')"><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg></button></li>'
                    else:
                         summary_html += f"<li>{advice_text}</li>"
            if in_list: summary_html += '</ul>'

        return render_template_string(RESULTS_TEMPLATE, summary_html=summary_html, original_transcript=transcript, audio_cues_enabled=preferences.get('audio_cues') == 'yes')
   
    return redirect(url_for('home'))

# --- Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True)

from gtts import gTTS
import os

# 1️⃣ Sentences per language
english_sentences = [
    "The weather is perfect for a morning walk.",
    "I love reading books by the riverside.",
    "Can you help me find the nearest coffee shop?",
    "Technology is changing the world every day.",
    "Our team won the hackathon with this project."
]

hindi_sentences = [
    "सुबह की सैर के लिए मौसम बिल्कुल सही है।",
    "मुझे नदी किनारे किताबें पढ़ना पसंद है।",
    "क्या आप मुझे सबसे नज़दीकी कॉफी शॉप दिखा सकते हैं?",
    "प्रौद्योगिकी हर दिन दुनिया को बदल रही है।",
    "हमारी टीम ने इस प्रोजेक्ट से हैकथॉन जीत लिया।"
]

tamil_sentences = [
    "காலை நடைப்பயிற்சிக்கு வானிலை சரியானது.",
    "நदीக்கரையில் புத்தகங்களை படிப்பது எனக்கு பிடிக்கும்.",
    "சிறந்த காபி கடையை கண்டுபிடிக்க உதவ முடியுமா?",
    "தொழில்நுட்பம் ஒவ்வொரு நாளும் உலகத்தை மாற்றுகிறது.",
    "எங்கள் குழு இந்த திட்டத்துடன் ஹாகதான் வென்றது."
]

# 2️⃣ Output folders
output_folder = "data/fake"
os.makedirs(output_folder, exist_ok=True)

# 3️⃣ Function to generate clips
def generate_clips(sentences, lang, prefix):
    for i, sentence in enumerate(sentences, 1):
        tts = gTTS(text=sentence, lang=lang)
        filename = os.path.join(output_folder, f"{prefix}_{i}.wav")
        tts.save(filename)
        print(f"Saved: {filename}")

# 4️⃣ Generate clips
generate_clips(english_sentences, "en", "english_fake")
generate_clips(hindi_sentences, "hi", "hindi_fake")
generate_clips(tamil_sentences, "ta", "tamil_fake")

print("All fake clips generated!")

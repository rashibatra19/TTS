import requests
from bs4 import BeautifulSoup
import re
import csv
import os
import time 
import pyttsx3
# Function to extract sentences from a given URL
def extract_sentences(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Assuming articles are within paragraph tags
        paragraphs = soup.find_all('p')
        sentences = []

        for paragraph in paragraphs:
            sentences.extend(paragraph.text.split('. '))  # Split sentences based on period
        return sentences
    except Exception as e:
        print(f"Error while processing URL {url}: {e}")
        return []

# Function to clean and preprocess the sentences
def preprocess_sentences(sentences, keywords):
    cleaned_sentences = []
    
    for sentence in sentences:
        # Step 1: Strip whitespace and remove empty strings
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Step 2: Remove URLs
        sentence = re.sub(r'https?://\S+', '', sentence)
        
        # Step 3: Remove unwanted characters (like HTML entities)
        sentence = re.sub(r'[^\w\s.,!?\'\"-]', '', sentence)
        
        # Step 4: Convert to lowercase for matching
        lowercase_sentence = sentence.lower()

        # Step 5: Check for relevant technical content based on keywords
        if any(keyword.lower() in lowercase_sentence for keyword in keywords):
            cleaned_sentences.append(sentence)

    return cleaned_sentences

def preprocess_for_tts(text):
    # TTS-related keywords
    text = text.replace("API", "A P I")
    text = text.replace("APIs", "A P I s")
    text = text.replace("apis", "A P I s")
    text = text.replace("SDK", "S D K")
    text = text.replace("sdk", "S D K")
    text = text.replace("TTS", "T T S")
    text = text.replace("tts", "T T S")
    text = text.replace("OCR", "O C R")
    text = text.replace("ocr", "O C R")
    text = text.replace("Twilio", "Twi-lee-oh")
    text = text.replace("text-to-speech", "text to speech")
    text = text.replace("speech synthesis", "speech sin-thuh-sis")
    text = text.replace("speech generation", "speech jen-uh-ray-shun")
    text = text.replace("voice synthesis", "voice sin-thuh-sis")
    text = text.replace("neural TTS", "neural T T S")
    text = text.replace("prosody", "prah-suh-dee")
    text = text.replace("phonemes", "foh-neems")
    text = text.replace("speech waveform", "speech wave-form")
    text = text.replace("acoustic model", "uh-koo-stik model")

    # API and SDK-related keywords
    text = text.replace("REST API", "R E S T A P I")
    text = text.replace("HTTP request", "H T T P request")
    text = text.replace("Webhooks", "web-hooks")
    text = text.replace("OAuth", "Oh-Auth")
    text = text.replace("JSON", "Jay-son")
    text = text.replace("XML", "X M L")
    text = text.replace("API Gateway", "A P I Gateway")
    text = text.replace("CUDA", "koo-duh")
    text = text.replace("cuda", "koo-duh")

    # Cloud platforms
    text = text.replace("AWS Polly", "A W S Polly")
    text = text.replace("Google Text-to-Speech", "Google Text to Speech")
    text = text.replace("Azure Cognitive Services", "Azure Cognitive Services")
    text = text.replace("IBM Watson TTS", "I B M Watson T T S")
    text = text.replace("Google Cloud", "Google Cloud")
    text = text.replace("Microsoft Azure", "Microsoft Azure")
    text = text.replace("AWS", "A W S")
    text = text.replace("Cloud-based services", "Cloud-based services")

    # AI and Machine Learning
    text = text.replace("Neural networks", "neural networks")
    text = text.replace("Deep learning", "deep learning")
    text = text.replace("Transformer models", "transformer models")
    text = text.replace("BERT", "B E R T")
    text = text.replace("GPT", "G P T")
    text = text.replace("Fine-tuning", "fine tuning")
    text = text.replace("Pretrained models", "pre-trained models")
    text = text.replace("Sequence-to-sequence", "sequence to sequence")
    text = text.replace("Training dataset", "training data set")
    text = text.replace("Speech model", "speech model")

    # Programming and tools
    text = text.replace("Python", "Python")
    text = text.replace("TensorFlow", "Ten-sor-flow")
    text = text.replace("PyTorch", "Pie-torch")
    text = text.replace("Hugging Face", "Hugging Face")
    text = text.replace("Flask", "Flask")
    text = text.replace("FastAPI", "Fast A P I")
    text = text.replace("JavaScript", "Java-Script")
    text = text.replace("Node.js", "Node J S")
    text = text.replace("NLP", "N L P")
    text = text.replace("NLP pipeline", "N L P pipeline")

    # Dataset and preprocessing
    text = text.replace("Dataset", "data set")
    text = text.replace("Annotation", "annotation")
    text = text.replace("Training data", "training data")
    text = text.replace("Preprocessing", "pre-processing")
    text = text.replace("Tokenization", "toh-kuh-nuh-zay-shun")
    text = text.replace("Phonetics", "fuh-ne-tiks")
    text = text.replace("Feature extraction", "feature extraction")
    text = text.replace("Data augmentation", "data augmentation")
    text = text.replace("Spectrogram", "spek-truh-gram")
    text = text.replace("Alignment", "alignment")

    # Additional cases for better pronunciation
    text = text.replace("CLI", "C L I")
    text = text.replace("URL", "U R L")
    text = text.replace("SQL", "S Q L")
    text = text.replace("CSS", "C S S")
    text = text.replace("HTML", "H T M L")
    text = text.replace("IP address", "I P address")
    text = text.replace("GitHub", "Git-Hub")
    text = text.replace("Bluetooth", "Blue-tooth")
    text = text.replace("Linux", "Li-nux")
    text = text.replace("Wi-Fi", "Wi-Fi")
    text = text.replace("HTTP", "H T T P")
    text = text.replace("DNS", "D N S")

    text = text.replace("SQL", "S Q L")
    text = text.replace("NoSQL", "No-Sequel")
    text = text.replace("MySQL", "My-S-Q-L")
    text = text.replace("PostgreSQL", "Postgre-S-Q-L")
    text = text.replace("MongoDB", "Mon-go-D-B")

    # Programming Languages
    text = text.replace("C++", "C plus plus")
    text = text.replace("C#", "C sharp")
    text = text.replace("PHP", "P H P")
    text = text.replace("HTML", "H T M L")
    text = text.replace("CSS", "C S S")
    text = text.replace("SASS", "Sass")
    text = text.replace("SCSS", "S C S S")

    # DevOps Tools
    text = text.replace("CI/CD", "C I / C D")
    text = text.replace("Kubernetes", "Koo-ber-net-ees")
    text = text.replace("Docker", "Docker")
    text = text.replace("YAML", "Yam-ul")
    text = text.replace("Ansible", "An-si-bul")

    # Security-related
    text = text.replace("SSL", "S S L")
    text = text.replace("TLS", "T L S")
    text = text.replace("JWT", "J W T")

    # Cloud and Platform-related
    text = text.replace("GCP", "G C P")
    text = text.replace("Azure", "Azure")
    text = text.replace("IAM", "I A M")
    text = text.replace("EC2", "E C 2")
    text = text.replace("S3", "S 3")
    text = text.replace("Lambda", "Lam-da")
    text = text.replace("VPC", "V P C")

    # AI and NLP Terms
    text = text.replace("LSTM", "L S T M")
    text = text.replace("GRU", "G R U")
    text = text.replace("RNN", "R N N")
    text = text.replace("CNN", "C N N")
    text = text.replace("GAN", "G A N")

    # Frameworks and Tools
    text = text.replace("Jupyter", "Ju-pi-ter")
    text = text.replace("Keras", "Keh-ras")
    text = text.replace("NumPy", "Num-Pie")
    text = text.replace("SciPy", "Sigh-Pie")
    text = text.replace("Matplotlib", "Mat-plot-lib")

    # Additional Pronunciation Considerations
    text = text.replace("2024", "twenty twenty-four")
    text = text.replace("404", "four oh four")
    text = text.replace("200", "two hundred")

    # Units of Measurement
    text = text.replace("MB", "megabyte")
    text = text.replace("GB", "gigabyte")
    text = text.replace("TB", "terabyte")
    text = text.replace("MHz", "megahertz")
    text = text.replace("GHz", "gigahertz")

    # UI/UX Terms
    text = text.replace("UX", "user experience")
    text = text.replace("UI", "user interface")
    text = text.replace("IoT", "I O T")
    text = text.replace("SaaS", "Sass")
    text = text.replace("PaaS", "Pass")
    text = text.replace("IaaS", "E-ass")
    text = text.replace("CRUD", "crud")
    text = text.replace("ORM", "O R M")
    text = text.replace("SOLID", "solid")
    text = text.replace("ACID", "acid")
    text = text.replace("gRPC", "g R P C")
    return text

# Function to save sentences into a CSV file
def save_to_csv(data, filename):
    # Open the file in write mode
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['URL', 'Sentence'])
        
        # Write the data
        for url, sentence in data:
            writer.writerow([url, sentence])

# Function to convert text to speech using gTTS
# def text_to_speech(text, filename):
#     for attempt in range(5):  # Retry up to 5 times
#         try:
#             tts = gTTS(text=text, lang='en')
#             tts.save(filename)
#             break  # Exit loop if successful
#         except gTTSError as e:
#             if '429' in str(e):
#                 print("Too many requests. Retrying...")
#                 time.sleep(5)  # Wait before retrying
#             else:
#                 print(f"Error occurred: {e}")
#                 break  # Exit loop on other errors

# def text_to_speech(text, filename):
#     # Initialize TTS
#     tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # Example model; adjust as necessary
#     try:
#         tts.tts_to_file(text=text, file_path=filename)
#     except Exception as e:
#         print(f"Error occurred while converting text to speech: {e}")

def text_to_speech(text, filename):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties before adding anything to speak
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    # Save the speech to a file
    engine.save_to_file(text, filename)

    # Run the speech engine
    engine.runAndWait()


# List of URLs to process
urls = [
    'https://stackoverflow.com/questions/53939383/testing-text-to-speech-tts-in-browser',
    'https://www.twilio.com/blog/2011/08/testing-twilios-text-to-speech-engine-using-twilio-client.html',
    'https://en.wikipedia.org/wiki/Software_development_kit',
    'https://docs.coqui.ai/en/latest/',
    'https://www.postman.com/api-platform/api-documentation/#:~:text=API%20documentation%20is%20a%20set,of%20common%20requests%20and%20responses.',
    'https://docs.nvidia.com/cuda/',
    'https://docs.aws.amazon.com/',
    'https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods',
    'https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model#:~:text=BERT%20is%20designed%20to%20help,%2Dand%2Danswer%20data%20sets.',
    'https://en.wikipedia.org/wiki/Natural_language_processing',
    'https://en.wikipedia.org/wiki/XML',
    'https://stackoverflow.blog/2022/06/02/a-beginners-guide-to-json-the-data-format-for-the-internet/',
    'https://medium.com/@SahanaGhosh8/real-time-gcp-google-cloud-platform-interview-questions-and-candidate-a049ea5a3348',
    'https://restfulapi.net/',
    'https://en.wikipedia.org/wiki/Hugging_Face',
    'https://en.wikipedia.org/wiki/IBM',
    'https://cloud.google.com/text-to-speech/docs',
    'https://www.databricks.com/glossary/what-is-dataset',
    'https://en.wikipedia.org/wiki/Data_augmentation',
    'https://en.wikipedia.org/wiki/Phonetics',
    'https://fastapi.tiangolo.com/',
    'https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-tokenization',
    'https://en.wikipedia.org/wiki/C%2B%2B',
    'https://en.wikipedia.org/wiki/SQL',
    'https://en.wikipedia.org/wiki/OAuth',
    # 'https://www.techtarget.com/iotagenda/definition/Internet-of-Things-IoT',

]

# Extended list of keywords for a more comprehensive technical dataset
keywords = [
    'API', 'TTS', 'SDK', 'CUDA' 'Twilio', 'text-to-speech', 'speech synthesis', 'speech generation', 
    'voice synthesis', 'neural tts', 'prosody', 'phonemes', 'speech waveform', 'acoustic model',
    'REST API', 'HTTP request', 'Webhooks', 'OAuth', 'JSON', 'XML', 'Endpoints', 'API Gateway',
    'AWS Polly', 'Google Text-to-Speech', 'Azure Cognitive Services', 'IBM Watson TTS', 'Google Cloud', 
    'Microsoft Azure', 'AWS', 'Cloud-based services',
    'Neural networks', 'Deep learning', 'Transformer models', 'BERT', 'GPT', 'Fine-tuning', 
    'Pretrained models', 'Sequence-to-sequence', 'Training dataset', 'Speech model',
    'Python', 'TensorFlow', 'PyTorch', 'Hugging Face', 'Flask', 'FastAPI', 'JavaScript', 'Node.js',
    'NLP', 'NLP pipeline', 'Dataset', 'Annotation', 'Training data', 'Preprocessing', 
    'Tokenization', 'Phonetics', 'Feature extraction', 'Data augmentation', 'Spectrogram', 'Alignment',
    'SQL', 'NoSQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'C++', 'C#', 'PHP', 'HTML', 'CSS',
    'SASS', 'SCSS', 'CI/CD', 'Kubernetes', 'Docker', 'YAML', 'Ansible', 'SSL', 'TLS', 'JWT',
    'GCP', 'IAM', 'EC2', 'S3', 'Lambda', 'VPC', 'LSTM', 'GRU', 'RNN', 'CNN', 'GAN', 
    'Jupyter', 'Keras', 'NumPy', 'SciPy', 'Matplotlib','UI','UX','IoT', 'SaaS', 'PaaS', 'IaaS', 'CRUD', 'ORM', 'SOLID', 'ACID', 'gRPC','MB','GB','TB',
    'MHz','GHz'
]

# Store cleaned sentences with their URLs
cleaned_sentences_with_urls = []

# Extract sentences from each URL and process them
# for url in urls:
#     sentences = extract_sentences(url)
#     cleaned_sentences = preprocess_sentences(sentences, keywords)

#     # Append the cleaned sentences with the corresponding URL
#     for sentence in cleaned_sentences:
#         cleaned_sentences_with_urls.append((url, sentence))

# # Save cleaned sentences to a CSV file
# csv_filename = 'cleaned_sentences.csv'
# save_to_csv(cleaned_sentences_with_urls, csv_filename)

# # Create a directory for audio clips
# os.makedirs('clips', exist_ok=True)

# # Convert cleaned sentences to speech and save audio files
# for url, sentence in cleaned_sentences_with_urls:
#     # Preprocess for TTS
#     processed_sentence = preprocess_for_tts(sentence)

#     # Create a unique filename for each sentence
#     sanitized_sentence = re.sub(r'[<>:"/\\|?*]', '', processed_sentence[:30])  # Remove invalid characters
#     audio_filename = f'clips/{sanitized_sentence}.mp3'

#     # Generate audio file
#     text_to_speech(processed_sentence, audio_filename)

os.makedirs('your_dataset/clips', exist_ok=True)

# Prepare metadata for Coqui AI TTS
metadata = []

# Read the existing CSV file with cleaned sentences and URLs
with open('cleaned_sentences.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    
    for i, row in enumerate(reader):
        url, sentence = row
        
        # Preprocess for TTS
        processed_sentence = preprocess_for_tts(sentence)

        # Create a unique filename for each sentence
        audio_filename = f'audio_{i+1}.wav'
        audio_path = os.path.join('your_dataset/clips', audio_filename)

        # Generate audio file
        text_to_speech(processed_sentence, audio_path)

        # Add metadata
        metadata.append(f'clips/{audio_filename}|{processed_sentence}')

# Save metadata to CSV
with open('your_dataset/metadata.csv', 'w', newline='', encoding='utf-8') as f:
    for line in metadata:
        f.write(line + '\n')

print("Dataset preparation complete. Check the 'your_dataset' folder for the result.")
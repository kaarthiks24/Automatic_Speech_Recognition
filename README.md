# Automatic_Speech_Recognition
This repository is used for Real time speech recognition to transcribe the voice/audio into text and chat with it

• Engineered end-to-end solution incorporating WhisperX OpenAI model for real-time voice-to-text conversion and
diarization for speaker segmentation
• Integrated ChromaDB backend for efficient data storage and retrieval, and OpenAI GPT for RAG chatbot functionality
• Developed intuitive chatbot interface using Gradio frontend, enabling seamless access to conversation transcripts

This application uses an open source python library called WhisperX. Install using the command:
```
pip install git+https://github.com/m-bain/whisperx.git
```
Whisper uses ffmpeg. Hence it also requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:
```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## To install dependencies
```
pip install requirements.txt
```

## Run the gradio application
```
gradio voice.py
```

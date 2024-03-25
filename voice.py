import torch
import whisper
import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperx import load_align_model, align
from whisperx.diarize import assign_word_speakers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import gradio as gr

from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

import tempfile
import os




def transcribe(inputs):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    script = model.transcribe(inputs)

    diarization_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN)
    diarized = diarization_pipeline(inputs)



    model_a, metadata = load_align_model(language_code=script["language"], device="cpu")
    script_aligned = align(script["segments"], model_a, metadata, audio_path, "cpu")

# Align Speakers
    result_segments, word_seg = list(assign_word_speakers(
        diarized, script_aligned
    ).values())
    transcribed = []
    for result_segment in result_segments:
        transcribed.append(
            {
                "start": result_segment["start"],
                "end": result_segment["end"],
                "text": result_segment["text"],
                "speaker": result_segment["speaker"],
            }
        )
    srt_file_path = "subtitles7.srt"
    with open(srt_file_path, 'w') as srt_file:
        subtitles = ""
        count = 1  # Initialize subtitle count

        for entry in transcribed:
            start_time = entry["start"]
            end_time = entry["end"]

            speaker = entry["speaker"]
            speaker = {
                "SPEAKER_00": "Support",
                "SPEAKER_01": "Customer",
                "SPEAKER_02": "Third",
                "SPEAKER_04": "Fourth"
            }[speaker]

            text = speaker + ": " + entry["text"]

        # Convert times to the SubRip format (hours:minutes:seconds,milliseconds)
            start_time_srt = '{:02}:{:02}:{:06.3f}'.format(int(start_time // 3600), int((start_time % 3600) // 60), start_time % 60)
            end_time_srt = '{:02}:{:02}:{:06.3f}'.format(int(end_time // 3600), int((end_time % 3600) // 60), end_time % 60)

        # Write the subtitle entry to the .srt file
            srt_file.write(str(count) + '\n')
            srt_file.write(start_time_srt + ' --> ' + end_time_srt + '\n')
            srt_file.write(text + '\n\n')
            subtitles += str(count) + '\n'
            subtitles += start_time_srt + ' --> ' + end_time_srt + '\n'
            subtitles += text + '\n\n'

            count += 1 
    return subtitles
    

MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8


device = 0 if torch.cuda.is_available() else "cpu"



HF_TOKEN="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"
audio_path="/Users/kaarthik/LLM/whisp/test.mp3"




device = "cpu" 
audio_file = "/Users/kaarthik/LLM/whisp/Learn English.mp3"
batch_size = 16 
compute_type = "int8" 

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
# result = model.transcribe(audio, batch_size=batch_size)
    

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

template = """You are a chatbot who helps to understand the conversation between customer and support person. You have the transcription of the conversation with you where the template of it is:
Example of a transcription where the customer speaks-

4 (indicating the number of conversation)
00:00:14.003 --> 00:00:15.500 (indicating the time stamp)
Customer:  and more people making stuff. (indicating who speaks the message: dialogue uttered)

Another example where a Support person speaks here-
5
00:00:19.644 --> 00:00:20.309
Support:  Ah, yeah.

Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def retrieve(inputs,history):
    model = ChatOpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db=Chroma(persist_directory="/Users/kaarthik/LLM/whisp/DIR", embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 1})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    ans=chain.invoke(inputs)
    return ans


demo = gr.Blocks()
Chatbot=gr.Chatbot(height=500,label="Transciption")
qanda = gr.ChatInterface(
    fn=retrieve,
    fill_height=True,
    chatbot=Chatbot,
    
        # inputs="text",
    # outputs="text",
    # theme="huggingface",
    title="Q and A on Transcripts",
    description=(
        "Ask questions about the the transcripts"
    ),
)


mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
    ],
    outputs="text",
    theme="huggingface",
    title="Voice to Text using Whisper",
    description=(
        "Speak in the microphone to convert Voice -> Text using Whisper - OpenAI"
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["upload"], type="filepath", label="Audio file"),
    ],
    outputs="text",
    theme="huggingface",
    title="Voice to Text using Whisper",
    description=("Upload a file to convert Audio -> Text using Whisper - OpenAI"),
    allow_flagging="never",
)


with demo:
    gr.TabbedInterface([mf_transcribe, file_transcribe, qanda], ["Microphone", "Audio file", "QA"])

demo.launch()


import pyaudio
import wave
import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch
import pyttsx3
import os
from flet import *

# إعداد المتغيرات
history = []
audio_buffer = []
model_path = "models"

# تحميل النماذج
def load_models():
    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_names = ["whisper-base", "t5-small", "distilbert-base-uncased-finetuned-sst-2-english"]
        for model_name in model_names:
            model_file_path = os.path.join(model_path, model_name)
            if not os.path.exists(model_file_path):
                print(f"جاري تحميل النموذج {model_name}...")
                if model_name == "whisper-base":
                    whisper.load_model(model_name)
                elif model_name == "t5-small":
                    T5ForConditionalGeneration.from_pretrained(model_name)
                    T5Tokenizer.from_pretrained(model_name)
                else:
                    pipeline("sentiment-analysis", model=model_name)
                print(f"تم تحميل النموذج {model_name}")
            else:
                print(f"النموذج {model_name} موجود مسبقًا")
    except Exception as e:
        print(f"حدث خطأ أثناء تحميل النماذج: {e}")

# تحميل النماذج تلقائيًا
load_models()

# تحميل النماذج للاستخدام
try:
    model = whisper.load_model("base")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dipsik_model = pipeline("sentiment-analysis")
except Exception as e:
    print(f"حدث خطأ أثناء تحميل النماذج: {e}")

# تسجيل الصوت
def record_audio():
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        print("تسجيل الصوت...")
        audio_buffer = []
        while True:
            data = stream.read(1024)
            audio_buffer.append(data)
            if len(audio_buffer) > 100:  # تسجيل لمدة 5 ثوان
                break
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("تم تسجيل الصوت")
        return audio_buffer
    except Exception as e:
        print(f"حدث خطأ أثناء تسجيل الصوت: {e}")

# معالجة الصوت
def transcribe_audio(audio_buffer):
    try:
        with wave.open("audio_record.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b''.join(audio_buffer))
        result = model.transcribe("audio_record.wav")
        user_text = result["text"]
        history.append({"role": "user", "content": user_text})
        return user_text
    except Exception as e:
        print(f"حدث خطأ أثناء معالجة الصوت: {e}")

# تحليل المشاعر
def analyze_sentiment(text):
    try:
        sentiment = dipsik_model(text)
        return sentiment
    except Exception as e:
        print(f"حدث خطأ أثناء تحليل المشاعر: {e}")

# توليد الاستجابة
def generate_response():
    try:
        input_text = history[-1]["content"]
        input_ids = t5_tokenizer.encode("generate response to: " + input_text, return_tensors="pt")
        output = t5_model.generate(input_ids, max_length=100)
        response = t5_tokenizer.decode(output[0], skip_special_tokens=True)
        history.append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        print(f"حدث خطأ أثناء توليد الاستجابة: {e}")

# التحدث مع المستخدم
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"حدث خطأ أثناء التحدث مع المستخدم: {e}")

def main(page: Page):
    page.title = "تطبيق المحادثة"
    page.horizontal_alignment = "center"

    def record_and_respond(e):
        audio_buffer = record_audio()
        user_text = transcribe_audio(audio_buffer)
        sentiment = analyze_sentiment(user_text)
        response = generate_response()
        speak(response)
        page.add(Text(f"المستخدم: {user_text}"))
        page.add(Text(f"الاستجابة: {response}"))
        page.update()

    page.add(ElevatedButton("تسجيل الصوت والاستجابة", on_click=record_and_respond))

if __name__ == "__main__":
    import flet as ft
    ft.app(target=main)

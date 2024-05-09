import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
import openai
from groq import Groq
from g4f.client import Client
import speech_recognition as sr
from pytube import YouTube
import tempfile
import os
from moviepy.editor import VideoFileClip


def get_video_id(url):
    if "youtu.be" in url:
        video_id = url.split("/")[-1]
        if "?" in video_id:
            video_id = video_id.split("?")[0]
    else:
        video_id = re.findall(r"v=(\w+)", url)[0]
    return video_id

def get_video_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'en'])
    script_text = ""
    for text_segment in transcript:
        script_text += text_segment['text'] + "\n"
    return script_text

def extract_audio(uploaded_file):
    with st.spinner("Извлечение аудио..."):
        # Сохранение загруженного файла во временную директорию
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Загрузка видеофайла с помощью moviepy
        video = VideoFileClip(temp_file_path)
        
        # Извлечение аудио из видео
        audio = video.audio
        
        # Сохранение аудио во временный файл
        audio_file_path = os.path.join(tempfile.gettempdir(), f"{uploaded_file.name}.wav")
        audio.write_audiofile(audio_file_path, codec='pcm_s16le')
        
        # Закрытие видео и аудио объектов
        video.close()
        audio.close()
        
        # Удаление временного файла
        os.unlink(temp_file_path)
    
    return audio_file_path

def audio_to_text(audio_file):
    with st.spinner("Транскрипция аудио..."):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language='ru-RU')
    return text

def process_with_nvidia(transcript_text, api_key, messages, temperature, top_p, max_tokens):
    openai.api_base = "https://integrate.api.nvidia.com/v1"
    openai.api_key = api_key
    
    if transcript_text:
        messages.append({"role": "user", "content": f"Пожалуйста, создай краткое содержание видео на русском языке, разделенное по темам. Используй форматирование, чтобы улучшить читаемость:\n\n{transcript_text}"})
    
    with st.spinner("Генерация краткого содержания (Nvidia)..."):
        completion = openai.ChatCompletion.create(
            model="meta/llama3-70b-instruct",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True
        )
        
        response_text = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.get("content") is not None:
                response_text += delta["content"]
    
    messages.append({"role": "assistant", "content": response_text})
    
    return response_text, messages

def process_with_groq(transcript_text, api_key, model, messages):
    client = Groq(api_key=api_key)
    
    if transcript_text:
        messages.append({"role": "user", "content": f"Пожалуйста, создай краткое содержание видео на русском языке, разделенное по темам. Используй форматирование, чтобы улучшить читаемость:\n\n{transcript_text}"})
    
    with st.spinner("Генерация краткого содержания (Groq)..."):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response_text = ""
        for chunk in completion:
            response_text += chunk.choices[0].delta.content or ""
    
    messages.append({"role": "assistant", "content": response_text})
    
    return response_text, messages

def process_with_free(transcript_text, messages):
    client = Client()
    
    if transcript_text:
        messages.append({"role": "user", "content": f"Пожалуйста, создай краткое содержание видео на русском языке, разделенное по темам. Используй форматирование, чтобы улучшить читаемость:\n\n{transcript_text}"})
    
    with st.spinner("Генерация краткого содержания (Free)..."):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        response_text = ""
        for chunk in completion:
            response_text += chunk.choices[0].delta.content or ""
    
    messages.append({"role": "assistant", "content": response_text})
    
    return response_text, messages

def process_question(question, messages, provider, api_key, temperature=None, top_p=None, max_tokens=None, model=None):
    if provider == "Nvidia":
        with st.spinner("Обработка вопроса (Nvidia)..."):
            if api_key:
                _, messages = process_with_nvidia("", api_key, messages, temperature, top_p, max_tokens)
            else:
                st.warning("Пожалуйста, введите API ключ для Nvidia.")
    elif provider == "Groq":
        with st.spinner("Обработка вопроса (Groq)..."):
            if api_key:
                _, messages = process_with_groq("", api_key, model, messages)
            else:
                st.warning("Пожалуйста, введите API ключ для Groq.")
    else:
        with st.spinner("Обработка вопроса (Free)..."):
            _, messages = process_with_free("", messages)
    
    return messages

def load_api_key(provider):
    return st.session_state.get(f"{provider}_api_key", "")

def save_api_key(provider, api_key):
    if f"{provider}_api_key" not in st.session_state:
        st.session_state[f"{provider}_api_key"] = api_key



def main():
    st.title("YouTube Video and Local Video Summary")
    
    st.sidebar.title("Настройки")
    provider = st.sidebar.selectbox("Поставщик", ["Free", "Nvidia", "Groq"], index=0)
    
    api_key = None
    if provider in ["Nvidia", "Groq"]:
        if f"{provider}_api_key" not in st.session_state:
            st.session_state[f"{provider}_api_key"] = load_api_key(provider)
        api_key = st.sidebar.text_input(f"API ключ для {provider}", type="password", value=st.session_state[f"{provider}_api_key"], key=f"{provider}_api_key")
        save_api_key(provider, api_key)

    
    if provider == "Nvidia":
        st.sidebar.subheader("Информация о модели")
        st.sidebar.write("Model ID: meta/llama3-70b-instruct")
        st.sidebar.write("Developer: Meta")
        st.sidebar.write("Context Window: 8,192 tokens")

        with st.sidebar.expander("Настройки модели"):
            temperature = st.slider("Температура", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
            max_tokens = st.number_input("Максимальное количество токенов", min_value=1, max_value=1024, value=1024, step=1)
    elif provider == "Groq":
        model = st.sidebar.selectbox("Модель", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b"])
        
        st.sidebar.subheader("Информация о модели")
        if model == "llama3-8b-8192":
            st.sidebar.write("Model ID: llama3-8b-8192")
            st.sidebar.write("Developer: Meta")
            st.sidebar.write("Context Window: 8,192 tokens")
        elif model == "llama3-70b-8192":
            st.sidebar.write("Model ID: llama3-70b-8192")
            st.sidebar.write("Developer: Meta")
            st.sidebar.write("Context Window: 8,192 tokens")
        elif model == "mixtral-8x7b-32768":
            st.sidebar.write("Model ID: mixtral-8x7b-32768")
            st.sidebar.write("Developer: Mistral")
            st.sidebar.write("Context Window: 32,768 tokens")
        elif model == "gemma-7b":
            st.sidebar.write("Model ID: gemma-7b")
            st.sidebar.write("Developer: Anthropic")
        
    video_url = st.text_input("Введите URL видео на YouTube:")
    uploaded_video = st.file_uploader("Или загрузите видео с вашего устройства:", type=["mp4", "avi", "mov"])
    
    if video_url or uploaded_video:
        if video_url:
            with st.spinner("Получение ID видео..."):
                video_id = get_video_id(video_url)
            with st.spinner("Получение транскрипта видео..."):
                transcript_text = get_video_transcript(video_id)
        else:
            audio_file = extract_audio(uploaded_video)
            transcript_text = audio_to_text(audio_file)

        
        messages = [{"role": "system", "content": f"Вы - помощник, который отвечает на вопросы по видео. Держите в уме текст видео и последние 5 вопросов пользователя."}]
        
        if provider == "Nvidia":
            if api_key:
                response_text, messages = process_with_nvidia(transcript_text, api_key, messages, temperature, top_p, max_tokens)
                st.subheader("Ответ от модели Nvidia:")
                st.markdown(response_text)
            else:
                st.warning("Пожалуйста, введите API ключ для Nvidia.")
        elif provider == "Groq":
            if api_key:
                response_text, messages = process_with_groq(transcript_text, api_key, model, messages)
                st.subheader("Ответ от модели Groq:")
                st.markdown(response_text)
            else:
                st.warning("Пожалуйста, введите API ключ для Groq.")
        else:
            response_text, messages = process_with_free(transcript_text, messages)
            st.subheader("Ответ от бесплатной модели:")
            st.markdown(response_text)

        st.subheader("Чат")

        chat_container = st.empty()
        question_container = st.empty()
        
        with question_container:
            question = st.text_input("Задайте уточняющий вопрос:")
        
        if st.button("Отправить"):
            if question:
                messages.append({"role": "user", "content": question})
                if provider == "Groq":
                    messages = process_question(question, messages[-5:], provider, api_key, model=model)
                elif provider == "Nvidia":
                    messages = process_question(question, messages[-5:], provider, api_key, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                else:
                    messages = process_question(question, messages[-5:], provider, api_key)
                
                chat_history = ""
                for message in messages[:-5]:
                    if message["role"] == "user":
                        chat_history += f"<p><strong>Пользователь:</strong> {message['content']}</p>"
                    else:
                        chat_history += f"<p><em>Ассистент:</em> {message['content']}</p>"
                
                chat_history += f"<p><strong>Пользователь:</strong> {question}</p>"
                chat_history += f"<p><em>Ассистент:</em> {messages[-1]['content']}</p>"
                
                chat_container.markdown(chat_history, unsafe_allow_html=True)
                
                # Удаление текущего поля ввода
                question_container.empty()
                
                # Создание нового чистого поля ввода
                with question_container:
                    st.text_input("Задайте уточняющий вопрос:", key=f"question_{len(messages)}")

if __name__ == "__main__":
    main()

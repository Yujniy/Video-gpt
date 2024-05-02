import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
import openai
from groq import Groq
from g4f.client import Client
import configparser
import os

CONFIG_FILE = "config.ini"

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

def process_with_nvidia(transcript_text, api_key, messages):
    openai.api_base = "https://integrate.api.nvidia.com/v1"
    openai.api_key = api_key
    
    if transcript_text:
        messages.append({"role": "user", "content": f"Пожалуйста сделай вижимку видео на русском языке по темам, применяй форматирование:\n\n{transcript_text}"})
    
    completion = openai.ChatCompletion.create(
        model="meta/llama3-70b-instruct",
        messages=messages,
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
    
    messages.append({"role": "assistant", "content": response_text})
    
    return response_text, messages

def process_with_groq(transcript_text, api_key, model, messages):
    client = Groq(api_key=api_key)
    
    if transcript_text:
        messages.append({"role": "user", "content": f"Пожалуйста сделай вижимку видео на русском языке по темам, применяй форматирование:\n\n{transcript_text}"})
    
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
        messages.append({"role": "user", "content": f"Пожалуйста сделай вижимку видео на русском языке по темам, применяй форматирование:\n\n{transcript_text}"})
    
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

def process_question(question, messages, provider, api_key, model=None):
    if provider == "Nvidia":
        if api_key:
            _, messages = process_with_nvidia("", api_key, messages)
        else:
            st.warning("Пожалуйста, введите API ключ для Nvidia.")
    elif provider == "Groq":
        if api_key:
            _, messages = process_with_groq("", api_key, model, messages)
        else:
            st.warning("Пожалуйста, введите API ключ для Groq.")
    else:
        _, messages = process_with_free("", messages)
    
    return messages


def load_api_key(provider):
    return st.session_state.get(f"{provider}_api_key", "")

def save_api_key(provider, api_key):
    if f"{provider}_api_key" not in st.session_state:
        st.session_state[f"{provider}_api_key"] = api_key



def main():
    st.title("YouTube Video Summary")
    
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
    
    if video_url:
        video_id = get_video_id(video_url)
        
        try:
            transcript_text = get_video_transcript(video_id)
            
            messages = [{"role": "system", "content": f"Вы - помощник, который отвечает на вопросы по видео. Держите в уме текст видео и последние 5 вопросов пользователя."}]
            
            if provider == "Nvidia":
                if api_key:
                    response_text, messages = process_with_nvidia(transcript_text, api_key, messages)
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
                        messages = process_question(question, messages[-5:], provider, api_key, model)
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

        except Exception as e:
            st.error(f"Не удалось получить текст. Ошибка: {str(e)}")

if __name__ == "__main__":
    main()

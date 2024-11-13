
import streamlit as st
import replicate
import os
import time  # 추가된 모듈

# App title
st.set_page_config(page_title="🤖 Chatbot")

# Markdown file path
MEMO_FILE_PATH = "responses.md"  # 답변이 저장된 마크다운 파일의 경로

# Replicate Credentials
with st.sidebar:
    st.title('🤖 Customer Chatbot')
    st.markdown('🔨 [Github Source](https://github.com/MinwooPark96/CSSC)')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Load responses from markdown file
def load_responses():
    responses = {}
    with open(MEMO_FILE_PATH, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_question = None
        current_answer = []
        
        for line in lines:
            if line.startswith("# "):  # 질문을 '#'로 구분
                if current_question:
                    responses[current_question.strip()] = ''.join(current_answer).strip()
                current_question = line[2:].strip()  # '#' 이후를 질문으로 사용
                current_answer = []
            else:
                current_answer.append(line)
        
        # 마지막 질문 저장
        if current_question:
            responses[current_question.strip()] = ''.join(current_answer).strip()
    
    return responses

# Store responses in session state
if "responses" not in st.session_state:
    st.session_state.responses = load_responses()

# Store chat messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Sentiment Analysis Toggle
with st.sidebar:
    # Checkbox for sentimental analysis with chat history clearing on toggle
    sentiment_analysis = st.checkbox("Turn on sentimental analysis", on_change=clear_chat_history)


def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    output = replicate.run(
        'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea',
        input={
            "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
            "temperature": 0.1, "top_p": 0.9, "max_length": 512, "repetition_penalty": 0
        }
    )
    return ''.join(output)

def get_response(user_input):
    responses = st.session_state.responses
    if user_input in responses:
        time.sleep(2)  
        return responses[user_input]
    else:
        return generate_llama2_response(user_input)

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
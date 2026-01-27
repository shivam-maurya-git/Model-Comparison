from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
st.markdown("### You can select models of your choice from dropdown")
st.header("Model Comparision")
model_num = st.number_input("Enter the number of models you want to compare [1-4]",min_value=1, max_value=4)
model_dict = {
    "zai-org":"zai-org/GLM-4.7-Flash", "deep-seek":"deepseek-ai/DeepSeek-V3.2", "meta-llama":"meta-llama/Llama-3.1-8B-Instruct",
    "OpenAI":"openai/gpt-oss-20b", "Qwen":"Qwen/Qwen3-8B", "Mistral":"mistralai/Mistral-7B-Instruct-v0.2",
    "Google" : "google/gemma-2-9b-it", "Xiaomi" : "XiaomiMiMo/MiMo-V2-Flash", "Nvidia" : "nvidia/OpenReasoning-Nemotron-7B",
    "MiniMaxAi" : "MiniMaxAI/MiniMax-M2.1", "moonshotai" : "moonshotai/Kimi-K2-Thinking", "Stabilityai" : "stabilityai/ar-stablelm-2-base"
}

selected_models = list()
if model_num>0:
    for i in range(model_num):
        choice = st.selectbox(f"{i+1}st model", list(model_dict.values()),key=f"sb_{i}")
        selected_models.append(choice)

user_input = st.text_input("Ask Your question : ")

# Just defining model invoke logic
def run_model(choice):
        llm = HuggingFaceEndpoint(repo_id = choice,task="text-generation")
        model = ChatHuggingFace(llm = llm)
        result = model.invoke(user_input)
        return result.content

results = [] # need to define because
# results is created only inside the if st.button(...) block

# If the button is not clicked, that block never runs

# Later loop still tries to read results → NameError

if st.button("Run all models"):
    with st.spinner("Running models in parallel..."):  #loading spinner in Streamlit UI
        with ThreadPoolExecutor(max_workers=model_num) as executor: #number of threads (usually = number of models #All calls start in parallel
            results = list(executor.map(run_model, selected_models)) 

for i, res in enumerate(results):
    st.success(f"Model {i+1}: {res}") #Display a success message :  model number and model output




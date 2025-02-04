
# prompt: now i want to deploy this in streamlit

!pip install -q transformers accelerate bitsandbytes huggingface_hub streamlit
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace with your model name (e.g., "deepseek-ai/deepseek-llm-7b-chat")
model_name = "deepseek-ai/deepseek-llm-7b-chat"

@st.cache_resource  # Cache the model and tokenizer to avoid reloading
def load_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # Auto-assigns layers to GPU/CPU
        torch_dtype=torch.float16,  # Use 16-bit precision
        load_in_8bit=True,          # Quantize to 8-bit (reduces VRAM)
        trust_remote_code=True      # Needed if the model uses custom code
    ).eval()  # Set to evaluation mode
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # Adjust for longer responses
        temperature=0.7,     # Control randomness (lower = more deterministic)
        top_p=0.9,           # Nucleus sampling
    )

    # Decode and return output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app
st.title("Chat with DeepSeek LLM")

user_input = st.text_input("Ask a question:")

if user_input:
    response = generate_response(user_input)
    st.text_area("Response:", value=response, height=200)
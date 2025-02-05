import streamlit as st
import nltk
import torch
from transformers import pipeline
from nltk.tokenize import word_tokenize

# Download necessary nltk data
nltk.download('punkt')

# Load a lightweight healthcare model (or use BioMistral if you have GPU)
try:
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    chatbot = pipeline("text-generation", model="BioMistral/BioMistral-7B", from_pt=True, device=device)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.warning("Falling back to a smaller model for compatibility.")
    chatbot = pipeline("text-generation", model="distilgpt2", device=device)

# Function for chatbot responses
def healthcare_chatbot(user_input):
    tokens = word_tokenize(user_input.lower())  # Tokenization
    if "symptom" in tokens:
        return "Please consult a doctor for an accurate diagnosis."
    elif "appointment" in tokens:
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in tokens:
        return "It's important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    else:
        response = chatbot(user_input, min_length=10, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']

# Streamlit UI
def main():
    st.title("Health Care Assistant Chatbot")
    st.write("Ask me anything related to healthcare.")

    user_input = st.text_area("How can I help you today?", height=150)
    if st.button("Submit"):
        if user_input:
            with st.spinner("Processing your query..."):
                response = healthcare_chatbot(user_input)
            st.success(f"Healthcare Assistant: {response}")
        else:
            st.warning("Please enter a message to continue.")

if __name__ == "__main__":
    main()

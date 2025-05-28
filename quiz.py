import streamlit as st
import time

st.title("Dementia Screening Quiz")
st.write("This quiz helps assess cognitive abilities. It is not a diagnosis but can indicate whether further evaluation is needed.")

st.info("**Instructions:** Read each question carefully and enter your response. Some questions require recalling information after a delay.")

# Questions List
questions = [
    "What is today's date (day, month, and year)?",
    "Can you name the current Prime Minister (or President) of your country?",
    "Where are you right now (home, city, or country)?",
    "Can you repeat this address after me: '15 Green Street, New York'? (Recall after 5 minutes)",
    "What is 100 minus 7? Now subtract 7 again, and continue until told to stop.",
    "Name three common objects I will show you (e.g., pen, watch, book) and recall them after 30 seconds.",
    "If you were lost in a new city, how would you find your way back home?",
    "Can you name at least five animals?",
    "Look at this picture (provide a simple image) and describe it.",
    "Can you draw a clock showing the time as 10:10?",
    "Can you recall what you ate for breakfast this morning?",
    "Can you follow this three-step command: 'Take this paper, fold it in half, and place it on the table'?",
    "Can you spell the word 'WORLD' backward?",
    "What would you do if you found a lost wallet on the street?",
    "Can you recognize and name a family member or friend in this photograph? (Show a familiar image)"
]

# Store user responses
responses = {}

# Iterate through questions
for i, question in enumerate(questions):
    responses[f'Q{i+1}'] = st.text_input(question, key=f'q{i+1}')
    
    # Delay recall questions
    if i == 3 or i == 5:
        st.write("Please remember this for later recall.")
        time.sleep(5)

# Submit Button
if st.button("Submit Quiz"):
    st.success("Quiz completed!")
    st.write("### Summary of Responses")
    for key, value in responses.items():
        st.write(f"**{key}:** {value}")
    
    st.write("If multiple responses show difficulty with recall, orientation, or problem-solving, consider seeking a professional cognitive assessment.")

st.warning("**Disclaimer:** This quiz is not a substitute for professional diagnosis. If cognitive decline is suspected, seek medical advice.")

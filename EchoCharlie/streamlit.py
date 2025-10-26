import streamlit as st

st.title("CharlieYaplin")


st.write("Hello from VS Code!")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
st.subheader("Choose from the following videos:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🖼️ Video 1"):
        st.session_state["choice"] = "img1"
    st.image("/Users/poojaravi/Documents/code/GitHub/wav2lip/poli/macron_1.png", caption="Option 1")

with col2:
    if st.button("🖼️ Video 2"):
        st.session_state["choice"] = "img2"
    st.image("/Users/poojaravi/Documents/code/GitHub/wav2lip/poli/obama_1.png", caption="Option 2")

with col3:
    if st.button("🖼️ Video 3"):
        st.session_state["choice"] = "img3"
    st.image("/Users/poojaravi/Documents/code/GitHub/wav2lip/poli/trudeau_1.png", caption="Option 3")

# --- Display different responses depending on which image was clicked ---
choice = st.session_state.get("choice", None)
path = ""
if choice == "img1":
    #st.write(" video.")
    path = "/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/videos/macron_1.mp4"
    st.video(path)

elif choice == "img2":
    path = "/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/videos/obama_1_one_word_error.mp4"
    st.video(path)

elif choice == "img3":
    #st.subheader("You clicked Image 3!")
    #st.write("Response for Image 3: You could run another model or show predictions.")
    path = "/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/videos/trudeau_1.mp4"
    st.video(path)

else:
    st.info("👆 Click one of the image buttons above to see the output.")


if st.button("Generate Audio"):
    get_path = path
    st.audio("/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/audio/output_trump.wav", format="audio/wav")
    

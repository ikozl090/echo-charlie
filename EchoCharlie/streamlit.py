import os
import streamlit as st
from echo_charlie import EchoCharlie
from echo_db import EchoDB

st.title("CharlieYaplin")

st.write("Hello from Itay, Pooja and Vishnou!")

echo_db = EchoDB(db_path="./demo_db_3", collection_name = "demo_collection_3", audio_db_name = "demo_audio_3.db")
main_path = "/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/"


st.subheader("Add reference videos to Database:")

uploaded_file = st.file_uploader("Upload any video file to Database", type=["mp4"])
#print(uploaded_file)
#echo_db.push_video(uploaded_file)

# Step 3: Create a local save directory
save_dir = "uploads/saved.mp4"

# Step 4: Construct the full local path
#local_path = os.path.join(save_dir, uploaded_file.name)

# Step 5: Save the uploaded file to disk
#with open(save_dir, "wb") as f:
    #f.write(uploaded_file.getbuffer())

col1, col2, col3, col4 = st.columns(4)
to_db = ""

with col1:
    if st.button("Add Trudeau to DB"):
        to_db = main_path + "videos/trudeau_ref.mp4"
        st.session_state["choice"] = "img1"
        echo_db.push_video(to_db)
        st.video(to_db)


with col2:
    if st.button("Add Obama to DB"):
        to_db = main_path + "videos/obama_ref.mp4"
        st.session_state["choice"] = "img2"
        echo_db.push_video(to_db)
        st.video(to_db)
        
with col3:
    if st.button("Add Macron to DB"):
        st.session_state["choice"] = "img3"
        to_db = main_path + "videos/macron_ref.mp4"
        echo_db.push_video(to_db)
        st.video(to_db)


with col4:
    if st.button("Add Trump to DB"):
        to_db = main_path + "videos/trump_ref.mp4"
        st.session_state["choice"] = "img4"
        echo_db.push_video(to_db)
        st.video(to_db)
    

#uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
st.subheader("Choose from the following videos:")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🖼️ Video 1"):
        st.session_state["choice"] = "img1"
    st.image(main_path + "st_display/trudeau_3.png", caption="Trudeau")

with col2:
    if st.button("🖼️ Video 2"):
        st.session_state["choice"] = "img2"
    st.image(main_path + "st_display/obama_3.png", caption="Obama")

with col3:
    if st.button("🖼️ Video 3"):
        st.session_state["choice"] = "img3"
    st.image(main_path + "st_display/macron_1.png", caption="Macron")

with col4:
    if st.button("🖼️ Video 4"):
        st.session_state["choice"] = "img4"
    st.image(main_path + "st_display/trump_1.png", caption="Option 3")


choice = st.session_state.get("choice", None)
path = ""
if choice == "img1":
    #st.write(" video.")
    path = main_path + "muted_videos/trudeau_3.mp4"
    st.video(path,muted=True)

elif choice == "img2":
    path = main_path + "muted_videos/obama_3_one_word_error.mp4"
    st.video(path,muted=True)

elif choice == "img3":
    #st.subheader("You clicked Image 3!")
    #st.write("Response for Image 3: You could run another model or show predictions.")
    path = main_path + "muted_videos/macron_1.mp4"
    st.video(path,muted=True)
    
elif choice == "img4":
    #st.subheader("You clicked Image 3!")
    #st.write("Response for Image 3: You could run another model or show predictions.")
    path = main_path + "muted_videos/trump_1.mp4"
    st.video(path,muted=True)

else:
    st.info("👆 Click one of the image buttons above to see the output.")


if st.button("Generate Audio"):
    get_path = path
    person = path.split("/")[-1].split(".")[0] + "_generated.wav"
    api_key = "bai-KexSTrjlCNGM_ZhBteIErVDK9saJ8NbbHsRInNm5YTIF4b4L"
    out = main_path + "audio/output_sample3.wav"
    transcripts = main_path + "transcripts/transcript.json"
    #echo_charlie = EchoCharlie(video_path=path,transcripts=transcripts,qwen_api_key=api_key,higgs_api_key=api_key)
    #_, aud = echo_charlie.forward(out_path=out)
    st.audio(main_path+"generated_audio/"+person, format="audio/wav")
    
if st.button("Generate Video"):
    person_vid = path.split("/")[-1].split(".")[0] + "_unmuted.mp4"
    st.video(main_path+"generated_videos/"+person_vid)

if st.button("Original Video"):
    person_vid = path.split("/")[-1].split(".")[0] + ".mp4"
    st.video(main_path+"videos/"+person_vid)
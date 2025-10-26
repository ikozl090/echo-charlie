# Echo Charlie

Echo Charlie is a multimodal voice conversion pipeline that turns a muted
video into a fully voiced clip. The system retrieves reference audio from an
indexed library, cleans up the lip-reading transcript, and then generates
new speech that matches the target speaker.

## 🎬 Demo

👉 [**View the EchoCharlie Demo on GitHub Pages**](https://ikozl090.github.io/echo-charlie/)

## Highlights
- Retrieval-augmented voice cloning using `EchoCharlie.echo_db.EchoDB`
- Visual speech recognition (lip reading) via `EchoCharlie.echo_vsr`
- Transcript clean-up with the Boson-hosted Qwen large language model
- Speech generation with the Boson Higgs audio model and reference audio
- Extensible building blocks for notebooks, Streamlit demos, and batch jobs

## Pipeline Overview
1. **Frame embedding** – `EchoCharlie.echo_frame.GetFrame` samples frames from
   the target video, runs DeepFace embeddings, and extracts audio if needed.
2. **Reference retrieval** – `EchoDB` stores embeddings in ChromaDB alongside
   an SQLite catalog of audio files. The closest reference clip is retrieved.
3. **Lip reading** – `EchoCharlie.echo_vsr.VSRInferencePipeline` runs the
   pre-trained LRS3 model to decode the video into raw text.
4. **Transcript correction** – `EchoCharlie.echo_qwen.QwenModel` fixes
   transcription errors and adds punctuation through a Boson API call.
5. **Voice synthesis** – `EchoCharlie.echo_higgs.HiggsModel` combines the
   cleaned transcript and reference audio to generate a new speech track.

## Repository Layout
- `EchoCharlie/` – Core Python package
  - `echo_charlie.py` – High-level orchestration class
  - `echo_db.py` – Reference media indexing and retrieval
  - `echo_frame.py` – Frame sampling, embeddings, and audio extraction
  - `echo_vsr.py` – Visual speech recognition inference wrapper
  - `echo_qwen.py` – Boson Qwen-based transcript cleaner
  - `echo_higgs.py` – Boson Higgs speech generation helper
  - `streamlit.py` – Prototype UI (uses hard-coded demo paths)
- `data/` – Sample videos, transcripts, embeddings, and generated audio
- `models/` – Pretrained VSR weights (`models/LRS3_V_WER19.1/…`)
- `nbs/` – Jupyter notebooks for experiments and tests
- `config.json` – Example configuration (contains placeholder API key)
- `main.py` – Minimal entry point stub

## Requirements
- Python 3.11
- FFmpeg (for video/audio I/O used by MoviePy and OpenCV)
- macOS with Apple Silicon is assumed by the default TensorFlow packages;
  for other platforms adjust the dependencies in `pyproject.toml`.
- Boson API access token with permission to call the Higgs and Qwen models

Install FFmpeg first, e.g. `brew install ffmpeg` on macOS.

## Installation
```bash
# Clone the repository
git clone <your fork>
cd echo-charlie

# (Recommended) create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -e .
```

This project also ships a `uv.lock` file. If you use `uv` you can run:

```bash
uv sync
```

The DeepFace backend downloads model weights the first time it runs. Allow
network access on first execution.

## Configuration
1. Create a `.env` or JSON file containing your Boson API key. **Do not commit
   real credentials.**
2. Pass the key into `EchoCharlie` when you instantiate it, or export
   `BOSON_API_KEY` and load it in your script.

Example `config.json`:
```json
{
  "api_key": "bai-REPLACE-ME"
}
```

## Quickstart
```python
from EchoCharlie import EchoCharlie

video_path = "data/videos/obama_1_one_word_error.mp4"
transcript_json = "data/transcripts/transcript.json"
output_audio = "data/generated_audio/obama_fixed.wav"
boson_key = "bai-REPLACE-ME"

references = [
    "data/videos/trump_ref.mp4",
    "data/videos/trudeau_ref.mp4",
    "data/videos/macron_ref.mp4",
    "data/videos/obama_ref.mp4",
]

charlie = EchoCharlie(
    video_path=video_path,
    transcripts=transcript_json,
    qwen_api_key=boson_key,
    higgs_api_key=boson_key,
    n_frames=1,
    emb_dim=128,
)

video, audio = charlie.forward(out_path=output_audio, references=references)
print(f"Generated speech saved to {audio}")
```

The reference videos will be added to the vector and audio databases on the
first run. Subsequent runs reuse the stored embeddings and metadata.

## Managing the Reference Database
- Use `EchoCharlie.echo_db.EchoDB.push_video` to ingest additional videos.
- Call `EchoDB.clear_db()` during development to wipe both ChromaDB and the
  SQLite catalog.
- Generated audio files are stored in the directory you pass to `forward`.

Run `python -m EchoCharlie.echo_db` during development to explore helper
methods, or inspect `EchoCharlie/streamlit.py` for a prototype UI. If you use
the Streamlit demo, replace the hard-coded paths with your own media folders.

## Notebooks
The `nbs/tests/*.ipynb` notebooks provide reproducible checks for ChromaDB
integration, import validation, and database behavior. Launch them with
`uv run jupyter lab` or your preferred notebook environment.

## Troubleshooting
- **Visual speech model errors** – Ensure the LRS3 model weights are present in
  `models/LRS3_V_WER19.1/`. Download them manually if necessary.
- **Audio extraction failures** – Confirm FFmpeg is installed and discoverable.
- **DeepFace embedding shape mismatch** – Make sure the `emb_dim` passed to
  `EchoCharlie` matches the embedding dimension of the selected DeepFace
  model (defaults to 128 for Facenet).
- **Boson API failures** – Verify your key has access to `Qwen3-32B` and
  `higgs-audio-generation` models and that network egress is allowed.


## Contributors


<table align="center">
<tr align="center">


<td width:25%>
Pooja Ravi

<p align="center">
<img src = "https://avatars3.githubusercontent.com/u/66198904?s=460&u=06bd3edde2858507e8c42569d76d61b3491243ad&v=4"  height="120" alt="Ansh Sharma">
</p>
<p align="center">
<a href = "https://github.com/01pooja10"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/pooja-ravi-9b88861b2/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td width:25%>
Itay Kozlov

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/110578288?v=4"  height="120" alt="Aneri Gandhi">
</p>
<p align="center">
<a href = "https://github.com/ikozl090"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/itay-kozlov/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td width:25%>
Vishnou Vinayagame

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/106386360?v=4"  height="120" alt="Aneri Gandhi">
</p>
<p align="center">
<a href = "https://github.com/vishnouvina"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/vinayagame-vishnou/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


</table>


## License
MIT © Itay Kozlov

This project is licensed under the MIT License - see the [License](LICENSE) file for details

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

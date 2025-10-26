import json
import numpy as np
from typing import List
import sys
import os

try:
    # Relative imports for when used as a package (e.g., from notebooks, other scripts)
    from .echo_frame import GetFrame
    from .echo_vsr import VSRInferencePipeline
    from .echo_qwen import QwenModel
    from .echo_db import EchoDB
    from .echo_embed import Embed
    from .echo_higgs import HiggsModel
except ImportError:
    # Absolute imports for when run as a script directly
    # Add the current directory to the Python path so we can import sibling modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from echo_frame import GetFrame
    from echo_vsr import VSRInferencePipeline
    from echo_qwen import QwenModel
    from echo_db import EchoDB
    from echo_embed import Embed
    from echo_higgs import HiggsModel
import warnings
warnings.filterwarnings("ignore")

class EchoCharlie(): 

    def __init__(self, video_path:str,
                       transcripts:str,
                       qwen_api_key:str, 
                       higgs_api_key:str,
                       n_frames:int=1,
                       emb_dim:int=128):
        
        super(EchoCharlie,self).__init__()
        
        self.video = video_path
        self.transcripts = transcripts
        
        self.GetFrame = GetFrame(n_frames, emb_dim)
        self.VSR = VSRInferencePipeline()
        self.Qwen = QwenModel(qwen_api_key)
        self.HiggsModel = HiggsModel(higgs_api_key)
        self.echo_db = EchoDB()
        self.Embed = Embed(emb_dim)
        
        
    def store_frames(self,refs:List):
        for ref in refs:
            self.echo_db.push_video(video_path=ref)
        
    def get_emb(self):
        emb,_,_ = self.GetFrame.forward(self.video)
        return emb
        
    def get_audio(self,embeddings:List[np.array]):
        return self.echo_db.get_audio_from_embedding(embeddings)
    
    def vsr(self):
        vsr_text = self.VSR.forward(self.video)
        cleaned_text = self.Qwen.qwen_out(vsr_text)
        return cleaned_text
    
    def retrieve_video(self,ref_video:str):
        ref_embedding, _ = self.get_frames(ref_video)
        return ref_embedding
    
    def forward(self,out_path:str,references:List[str]=None):
        if references is not None:
            self.store_frames(references)
        ref_emb = self.get_emb()
        ref_audio_dict = self.get_audio(ref_emb)[0][0]
        ref_audio = ref_audio_dict["path"]
        
        ref_transcript_key = ref_audio_dict["key"]
        with open(self.transcripts,"rb") as fl:
            transcripts = json.load(fl)
        print(ref_transcript_key, ref_audio)
        for videos in transcripts:
            if videos["video"] == ref_transcript_key+".mp4":
                ref_transcript = videos["transcript"]
        
        main_transcript = self.vsr()
        print(f"{main_transcript}")
        
        audio_path = self.HiggsModel.higgs_out(ref_audio,ref_transcript,main_transcript,out_path)
        
        return self.video, audio_path



def test():
    import json
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    api_key = config.get("api_key", "")

    v_pth = "data/videos/trudeau_3.mp4" #"data/videos/obama_3_one_word_error.mp4"
    transc = "data/transcripts/transcript.json"
    ec = EchoCharlie(video_path=v_pth,transcripts=transc,qwen_api_key=api_key,higgs_api_key=api_key)
    out = "data/audio/output_sample3.wav"
    refs = ["data/videos/trump_ref.mp4","data/videos/trudeau_ref.mp4","data/videos/macron_ref.mp4","data/videos/obama_ref.mp4"]
    v, a = ec.forward(out,refs)

if __name__ == "__main__":
    test()
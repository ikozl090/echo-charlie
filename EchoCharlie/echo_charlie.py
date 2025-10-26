import json
import numpy as np
from typing import List

from echo_frame import GetFrame
from echo_vsr import VSRInferencePipeline
from echo_qwen import QwenModel
from echo_db import EchoDB
from echo_embed import Embed
from echo_higgs import HiggsModel


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
        
    def store_frames(self):
        emb = EchoDB.push_video(self.video_path)
        return emb
        
    def get_audio(self,embeddings:List[np.array]):
        return EchoDB.get_audio_from_embedding(embeddings)
    
    def vsr(self):
        vsr_text = self.VSR.forward(self.video)
        cleaned_text = self.Qwen.qwen_out(vsr_text)
        return cleaned_text
    
    def retrieve_video(self,ref_video:str):
        ref_embedding, _ = self.get_frames(ref_video)
        return ref_embedding
    
    def forward(self,out_path:str):
        ref_emb = self.store_frames()
        ref_audio_dict = self.get_audio(ref_emb)
        ref_audio = ref_audio_dict["path"]
        
        ref_transcript_key = ref_audio_dict["key"]
        with open(self.transcripts,"rb") as fl:
            transcripts = json.load(fl)
            
        for videos in transcripts:
            if videos["video"] == ref_transcript_key:
                ref_transcript = videos["transcript"]
        
        main_transcript = self.vsr()
        audio_path = self.HiggsModel.higgs_out(ref_audio,ref_transcript,main_transcript,out_path)
        
        return self.video, audio_path
        
    
def test():
    api_key=""
    v_pth = "/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/videos/obama_3_one_word_error.mp4"
    transc = "/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/transcripts/transcript.json"
    ec = EchoCharlie(video_path=v_pth,transcripts=transc,qwen_api_key=api_key,higgs_api_key=api_key)
    out = "/Users/poojaravi/Documents/code/GitHub/echo-charlie/data/audio/output_sample.wav"
    v, a = ec.forward(out)

if __name__ == "__main__":
    test()
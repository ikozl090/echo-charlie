import cv2
import numpy as np
import matplotlib.pyplot as plt
from audio_extract import extract_audio
from echo_embed import Embed

class GetFrame:
    def __init__(self, n_frames=3, emb_dim=128):
        super(GetFrame,self).__init__()
        self.nframes = n_frames
        self.emb_dim = emb_dim
        self.emb = Embed(emb_dim)

    def parse_frames(self, video_path, st=10, end=60):
        vidcap = cv2.VideoCapture(video_path)
        mid = (st + end) / 2
        frames = []
        c = 0
        while True:
            c+=1
            ret, frame = vidcap.read()
            if c < st:
                continue
            elif ret and ((c % mid==0) or (c % st==0) or (c % end==0)):
                frames.append(frame)
            elif c > end:
                break
        assert len(frames) == self.nframes
        return frames
        
    def embed(self,img,model_name = "Facenet"):
        return self.emb.forward(img,model_name)

    def extract_audio(self,video_path, output_path):
        audio_fl = video_path.split("/")[-1].split(".")[0]
        extract_audio(video_path, output_path+audio_fl+".wav")
    
    def forward(self, video_path, out_audio_path):
        frames = self.parse_frames(video_path)
        self.extract_audio(video_path,out_audio_path)
        embeddings = np.zeros(self.nframes,self.emb_dim)
        for i in range(len(frames)):
            embeddings[i] = self.embed(frames[i])
        return embeddings
    
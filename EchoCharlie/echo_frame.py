import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
from .echo_embed import Embed

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
            elif ret and ((c == mid) or (c == st) or (c == end)):
                frames.append(frame)
            elif c > end:
                break
        print(f"DEBUG: Number of Frames is {len(frames)}")
        assert len(frames) == self.nframes
        return frames
        
    def embed(self,img,model_name = "Facenet"):
        return self.emb.forward(img,model_name)

    def extract_audio(self,video_path, output_path):
        audio_fl = video_path.split("/")[-1].split(".")[0]
        output_file = output_path+audio_fl+".wav"
        
        # Use moviepy to extract audio
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(output_file, verbose=False, logger=None)
            audio.close()
        video.close()
        
        return output_file
    
    # def forward(self, video_path, out_audio_path=None):
    #     key = video_path.split("/")[-1].split(".")[0]
    #     frames = self.parse_frames(video_path)
    #     audio_path = self.extract_audio(video_path,out_audio_path)
    #     embeddings = np.zeros(self.nframes,self.emb_dim)
    #     for i in range(len(frames)):
    #         embeddings[i] = self.embed(frames[i])
    #     return embeddings, audio_path, key
    

    def forward(self, video_path, out_audio_path=None):
        frames = self.parse_frames(video_path)
        key = video_path.split("/")[-1].split(".")[0]
        if out_audio_path is not None:
            audio_path = self.extract_audio(video_path,out_audio_path)
        else: audio_path = None
        embeddings = np.zeros(self.nframes,self.emb_dim)
        for i in range(len(frames)):
            embeddings[i] = self.embed(frames[i])
        return embeddings, audio_path, key
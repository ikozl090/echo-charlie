from deepface import DeepFace
import numpy as np
from numpy.linalg import norm

class Embed:
    def __init__(self, emb_dim=128):
        super(Embed,self).__init__()
        self.emb_dim = emb_dim
        
    def forward(self,image,model_name="Facenet"):
        out = DeepFace.represent(image, model_name = model_name)
        emb = out[0]["embedding"]
        emb = np.array(emb)
        assert emb.shape[0] == self.emb_dim
        return emb
    
    
    
    

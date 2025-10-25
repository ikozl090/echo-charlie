from deepface import DeepFace
import numpy as np
from numpy.linalg import norm

class Embed:
    def __init__(self, image, emb_dim=128):
        super(Embed,self).__init__()
        self.image = image
        self.emb_dim = emb_dim
        
    def forward(self,model_name="Facenet"):
        emb = DeepFace.represent(img_path = self.image, model_name = model_name)
        emb = np.array(emb)[np.newaxis,:]
        assert emb.shape[1] == self.emb_dim
        return emb
    
    

import chromadb
from chromadb import Collection

import numpy as np
import os 
import pickle
import numpy as np
from typing import List

class EchoDB(): 

    vdb_collection: Collection = None

    def __init__(self, db_path: str = "./echo_db", collection_name: str = "echo_collection"): 
        client = chromadb.PersistentClient(path=db_path) 
        self.vdb_collection = client.get_or_create_collection(name=collection_name)

        # TODO: Add functions for defining audio DB 

    def add_embedding_dir(self, embedding_dir: str, file_type: str = 'pkl'): 

        # NOTE: Only supports pickle files right now
        assert file_type == 'pkl'

        embedding_files = [os.path.splitext(f)[0] for f in os.listdir(embedding_dir) if os.path.isfile(os.path.join(embedding_dir, f)) and f.endswith('.pkl')]

        # Load data 
        embeddings = []
        for embedding_file in embedding_files:
            with open(f"{embedding_dir}/{embedding_file}.{file_type}", 'rb') as file: 
                embedding = pickle.load(file)

            embeddings.append(np.array(embedding))

        metadatas = [{"filename": f"{embedding_file}.png"} for embedding_file in embedding_files]

        self.vdb_collection.add(
            ids=embedding_files,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_vdb(self, embeddings: List[np.array]): 
        results = self.vdb_collection.query(
            query_embeddings=embeddings,
            n_results=3
        )

        return results

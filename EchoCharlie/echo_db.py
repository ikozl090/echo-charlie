import chromadb
from chromadb import Collection

import numpy as np
import os 
import pickle
import numpy as np
from typing import List
import sys

# pip install sqlite-utils mutagen
from sqlite_utils import Database
from mutagen import File as MutagenFile  # optional, to read tags/duration
from pathlib import Path
import hashlib, time

# EchoCharlie Modules 
try:
    from .echo_frame import GetFrame
except ImportError:
    # Add the current directory to the Python path so we can import sibling modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from echo_frame import GetFrame

class EchoDB(): 

    vdb_collection: Collection = None
    audio_db: Database = None 

    def __init__(self, db_path: str = "./echo_db", collection_name: str = "echo_collection", audio_db_name: str = "audio.db"): 
        super(EchoDB,self).__init__()
        # Initialize chromodb 
        client = chromadb.PersistentClient(path=db_path) 
        self.vdb_collection = client.get_or_create_collection(name=collection_name)

        # Initialize Audio DB - use absolute path to ensure consistency
        self.audio_db_path = os.path.abspath(audio_db_name)
        print(f"DEBUG: Initializing database at: {self.audio_db_path}")
        self.audio_db = Database(self.audio_db_path)

        # Create tables once (idempotent)
        self.audio_db["files"].create({
            "key": str,
            "path": str,
            "duration": float,
            "samplerate": int,
            "channels": int,
            "md5": str,
            "created_at": float
        }, pk="key", not_null={"key","path"}, if_not_exists=True)

        self.audio_db["tags"].create({"key": str, "tag": str}, pk=("key","tag"), if_not_exists=True)

    def push_video(self, video_path, n_frames: int = 1, audio_file_dir: str = 'audio_from_video'):
        gf = GetFrame(n_frames = n_frames)

        embeddings, audio_path, key = gf.forward(video_path, out_audio_path = audio_file_dir)
        
        self.add_embeddings(embeddings = embeddings, keys = [key for k in embeddings])
        self.index_audio(path = audio_path, key = key)
        

    def get_audio_from_embedding(self, embeddings: List[np.array]): 
        keys = self.query_vdb(embeddings) 
        results = []
        for key_list in keys: 
            key = self.__choose_key(key_list)
            result = self.query_audio(key = key)
            results.append(result)

        return results

    def __choose_key(self, keys: List[str]): 
        return keys[0]

    def load_embedding_dir(self, embedding_dir: str, file_type: str = 'pkl'): 
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

        return embeddings, embedding_files, metadatas 

    def add_embeddings(self, embeddings: List[np.array], keys: List[str], metadatas: List[dict] = None): 
        if metadatas: 
            self.vdb_collection.add(
                ids=keys,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else: 
            self.vdb_collection.add(
                ids=keys,
                embeddings=embeddings,
            )

    def add_embedding_dir(self, embedding_dir: str, file_type: str = 'pkl'): 

        embeddings, embedding_files, metadatas = self.load_embedding_dir(embedding_dir = embedding_dir, file_type = file_type)

        self.add_embeddings(embeddings = embeddings, keys = embedding_files, metadatas = metadatas)
        # self.vdb_collection.add(
        #     ids=embedding_files,
        #     embeddings=embeddings,
        #     metadatas=metadatas
        # )

    def query_vdb(self, embeddings: List[np.array]): 
        results = self.vdb_collection.query(
            query_embeddings=embeddings,
            n_results=3
        )

        ids = results['ids']

        return ids

    def md5sum(self, p: Path, chunk=1<<20):
        h = hashlib.md5()
        with p.open("rb") as f:
            while (b := f.read(chunk)):
                h.update(b)
        return h.hexdigest()

    def index_audio(self, key: str, path: str, tags=()):
        try:
            # Validate file exists
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            
            # Read audio metadata using mutagen first, fallback to scipy
            duration = None
            sr = None
            ch = None
            
            try:
                m = MutagenFile(path)
                if m and m.info:
                    duration = float(m.info.length) if hasattr(m.info, 'length') else None
                    sr = int(m.info.sample_rate) if hasattr(m.info, 'sample_rate') else None
                    ch = int(m.info.channels) if hasattr(m.info, 'channels') else None
            except:
                pass
            
            # Fallback: try scipy for WAV files
            if sr is None or duration is None:
                try:
                    import wave
                    with wave.open(str(p), 'rb') as wav_file:
                        sr = wav_file.getframerate()
                        ch = wav_file.getnchannels()
                        frames = wav_file.getnframes()
                        duration = frames / float(sr) if sr > 0 else None
                except:
                    pass

            # Insert into files table
            file_data = {
                "key": key,
                "path": str(p.resolve()),
                "duration": duration,
                "samplerate": sr,
                "channels": ch,
                "md5": self.md5sum(p),
                "created_at": time.time(),
            }
            
            # Debug: print what we're trying to insert
            print(f"DEBUG: Inserting file data: {file_data}")
            
            # Insert the file using raw SQL to ensure it works
            import sqlite3
            conn = sqlite3.connect(self.audio_db_path)
            cursor = conn.cursor()
            
            # Use INSERT OR REPLACE to match upsert behavior
            cursor.execute("""
                INSERT OR REPLACE INTO files (key, path, duration, samplerate, channels, md5, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_data['key'],
                file_data['path'],
                file_data['duration'],
                file_data['samplerate'],
                file_data['channels'],
                file_data['md5'],
                file_data['created_at']
            ))
            conn.commit()
            
            # Verify it was inserted
            cursor.execute("SELECT COUNT(*) FROM files WHERE key = ?", (key,))
            count = cursor.fetchone()[0]
            conn.close()
            
            print(f"✓ Inserted file: {key}")
            print(f"DEBUG: Files with key '{key}': {count}")

            # Insert tags
            for t in tags:
                self.audio_db["tags"].upsert({"key": key, "tag": t}, pk=("key","tag"))
            if tags:
                print(f"✓ Inserted {len(tags)} tags for {key}")
            
        except Exception as e:
            print(f"Error in index_audio: {e}")
            import traceback
            traceback.print_exc()
            raise

    def show_audio_db(self):
        """Display all content from the audio database."""
        print("\n📊 Audio Database Contents\n")
        print(f"  Database path: {self.audio_db_path}")
        
        # Get all files
        all_files = list(self.audio_db["files"].rows)
        
        if not all_files:
            print("  No audio files in database.")
            return
        
        # Display each file with its tags
        for file_row in all_files:
            key = file_row["key"]
            path = file_row["path"]
            duration = file_row["duration"]
            samplerate = file_row["samplerate"]
            channels = file_row["channels"]
            md5 = file_row["md5"]
            
            print(f"  Key: {key}")
            print(f"  Path: {path}")
            if duration:
                print(f"  Duration: {duration:.2f}s")
            if samplerate:
                print(f"  Sample Rate: {samplerate} Hz")
            if channels:
                print(f"  Channels: {channels}")
            print(f"  MD5: {md5}")
            
            # Get tags for this file
            tags = list(self.audio_db["tags"].rows_where("key = ?", [key]))
            if tags:
                tag_list = [tag["tag"] for tag in tags]
                print(f"  Tags: {', '.join(tag_list)}")
            else:
                print(f"  Tags: (none)")
            
            print()

        
    # ---- Query wrapper ----
    def query_audio(self, key=None, tag=None, min_duration=None, max_duration=None):
        """Flexible query interface for the audio DB."""
        clauses, params = [], {}

        if key:
            clauses.append("files.key = :key")
            params["key"] = key
        if tag:
            clauses.append("tags.tag = :tag")
            params["tag"] = tag
        if min_duration is not None:
            clauses.append("files.duration >= :min_dur")
            params["min_dur"] = min_duration
        if max_duration is not None:
            clauses.append("files.duration <= :max_dur")
            params["max_dur"] = max_duration

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        join = "JOIN tags USING(key)" if tag else ""
        sql = f"SELECT DISTINCT files.* FROM files {join} {where} ORDER BY created_at DESC"

        return list(self.audio_db.query(sql, params))

        

    

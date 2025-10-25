from openai import OpenAI
import base64
import os


class HiggsModel:
    def __init__(self,my_api_key,modalities=["text", "audio"],
                    max_completion_tokens=4096,
                    temperature=0.9,
                    top_p=0.95,
                    stream=False,
                    base_url="https://hackathon.boson.ai/v1"):
        
        self.modalities = modalities
        self.max_tokens = max_completion_tokens
        self.temp = temperature
        self.top_p = top_p
        self.stream = stream
        BOSON_API_KEY = my_api_key
        self.client = OpenAI(api_key=BOSON_API_KEY, base_url=base_url)
    
    
    def b64(self,path):
        return base64.b64encode(open(path, "rb").read()).decode("utf-8")

    def higgs_out(self,ref_audio,ref_transcript,out_transcript,output_path):
        
        system = (
            "You are an AI assistant designed to convert text into speech.\n"
            "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
            "<|scene_desc_start|>\nAudio is from a politician giving an energetic speech without pauses.\n<|scene_desc_end|>"
        )

        resp = self.client.chat.completions.create(
            model="higgs-audio-generation-Hackathon",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": ref_transcript},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": self.b64(ref_audio), "format": "wav"}
                    }],
                },
                {"role": "user", "content": out_transcript},
            ],
            modalities = self.modalities,
            max_completion_tokens = self.max_tokens,
            temperature=self.temp,
            top_p=self.top_p,
            stream=self.stream,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            extra_body={"top_k": 50},
        )

        audio_b64 = resp.choices[0].message.audio.data
        open(output_path, "wb").write(base64.b64decode(audio_b64))
        
        return output_path
from openai import OpenAI

class QwenModel:
    def __init__(self,my_api_key,modalities=["text", "audio"],
                    max_completion_tokens=2048,
                    temperature=0.0,
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
    
    def qwen_out(self, transcript_to_clean):
        
        system = (
            "This transcript comes from a lipreading machine learning model. Please correct only the words that are incorrect or out of place so the text reads as what the speaker most likely said. Keep the original content and structure — do not add or remove any words, only replace misheard ones. Also, convert all caps text into normal sentence casing and add punctuation where necessary.\n"
        )

        resp = self.client.chat.completions.create(
            model="Qwen3-32B-non-thinking-Hackathon",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": transcript_to_clean},
            ],
            modalities = self.modalities,
            max_completion_tokens = self.max_tokens,
            temperature=self.temp,
            top_p=self.top_p,
            stream=self.stream,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            extra_body={"top_k": 50},
        )

        return resp.choices[0].message.content
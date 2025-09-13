from openai import OpenAI
from .Model import Model

class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_keys[api_pos])

    def query(self, msg):
        try:
            # Check if the model is part of the o1 series
            if self.name in ['o1', 'o1-preview', 'o1-mini']:
                # Combine the system instruction with the user message
                combined_message = "You are a helpful assistant. " + msg
                messages = [{"role": "user", "content": combined_message}]
                # Use max_completion_tokens for o1 models
                completion = self.client.chat.completions.create(
                    model=self.name,
                    temperature=1.0,
                    
                    messages=messages,
                )
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ]
                # Use max_tokens for non-o1 models
                completion = self.client.chat.completions.create(
                    model=self.name,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    messages=messages,
                )
            response = completion.choices[0].message.content

        except Exception as e:
            print(e)
            response = ""

        return response
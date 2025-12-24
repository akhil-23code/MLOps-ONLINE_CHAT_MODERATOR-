from transformers import AutoConfig
import os

# Update this to your local path
MODEL_PATH = "models/champion_bert_model"
config = AutoConfig.from_pretrained(MODEL_PATH)

print("ID to Label Mapping in Config:")
print(config.id2label)


# uvicorn app.main:app --reload
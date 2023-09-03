from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

input = open(r'/Users/akil/work/hackathon/input.txt', 'r')
model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

tokenizer = AutoTokenizer.from_pretrained("DunnBC22/flan-t5-base-text_summarization_data")
model = AutoModelForSeq2SeqLM.from_pretrained("DunnBC22/flan-t5-base-text_summarization_data")
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


text = input.read()

def get_response(input_text):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=1024, return_tensors="pt").to(torch_device)
  gen_out = model.generate(**batch,max_length=120,min_length=30,num_beams=5, num_return_sequences=1, temperature=1.5)
  output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
  return output_text

output = get_response(text)
result = nlp(output)

print(output[0])
print("\nThe Sentiment of this summary: " + result[0]['label'])
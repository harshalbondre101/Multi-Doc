"""
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def generate_summary(important_parts, max_sentences=50):
    summaries = []
    for part in important_parts:
        parser = PlaintextParser.from_string(part, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count=max_sentences)  # Limit the number of sentences in the summary
        summaries.append(' '.join(str(sentence) for sentence in summary))

    combined_summary = ' '.join(summaries)
    return combined_summary


from transformers import T5ForConditionalGeneration, T5Tokenizer

def abstractive_summarization(text):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    if isinstance(text, list):
        combined_text = ' '.join(text)
    else:
        combined_text = text

    preprocess_text = "summarize: " + combined_text
    input_ids = tokenizer.encode(preprocess_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=8, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
"""

#------------------------ GPT ------------------------
import os
from openai import OpenAI
os.environ["OPENAI_API_KEY"] =  "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"

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("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"),
)

def summarize_text_with_openai(text, model="gpt-3.5-turbo"):
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [

            {
                "role": "user",
                "content": f"You are a summarizer. summarize the text given to you. You need to summarize only around the given text and do not add anything on your own. Summarize the following in 500 words: {text}",
            }
        ],
    }
    
    # Call the OpenAI API to generate the summary
    chat_completion = client.chat.completions.create(**payload)
    
    # Extract and return the generated summary
    summary = chat_completion.choices[0].message.content
    return summary



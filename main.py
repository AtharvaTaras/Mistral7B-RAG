import torch
import logging
import gradio as gr

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

# print(torch.cuda.is_available())

documents = SimpleDirectoryReader("mydata/").load_data()

mistral = LlamaCPP(
    model_url=None,
    model_path=r'A:\LMStudio-Models\TheBloke\Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_S.gguf', # if local model, use path instead of URL
    temperature=0.1,
    max_new_tokens=512,
    context_window=3900, # max is 4096
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="thenlper/gte-large"), embed_batch_size=4)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=mistral,
    embed_model=embed_model,
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

def fetch(myquery:str):
    output =  query_engine.query(myquery)
    #print(output)
    return output

root = gr.Interface(fn=fetch,
                    inputs='text',
                    outputs='text',
                    title='Mistral 7B RAG',
                    theme=gr.themes.Monochrome())

root.launch()
import torch
import PyPDF2 # pdf reader
import time
from pypdf import PdfReader
from io import BytesIO
from langchain.prompts import PromptTemplate # for custom prompt specification
from langchain.text_splitter import RecursiveCharacterTextSplitter # splitter for chunks
from langchain.embeddings import HuggingFaceEmbeddings # embeddings
from langchain.vectorstores import FAISS # vector store database

from langchain.chains import RetrievalQA # qa and retriever chain
from langchain.memory import ConversationBufferMemory # for model's memoy on past conversations
from langchain.document_loaders import PyPDFDirectoryLoader # loader fo files from firectory

from langchain.llms.huggingface_pipeline import HuggingFacePipeline # pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from accelerate import Accelerator
accelerator = Accelerator()

CHUNK_SIZE = 1000
# Using HuggingFaceEmbeddings with the chosen embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs = {"device": "cuda"})

# transformer model configuration
# this massively model's precision for memory effieciency
# The models accuacy is reduced.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tensor_1 = torch.rand (4,4)

model_id = "Deci/DeciLM-7B-instruct" # model repo id
device = 'cuda' # Run on gpu if available else run on cpu

#
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             device_map = "auto",
                                             quantization_config=quant_config)

# create a pipeline
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text = True,
                max_new_tokens=200,
                repetition_penalty = 1.1,
                num_beams=5,
                no_repeat_ngram_size=4,
                early_stopping=True)

llm = HuggingFacePipeline(pipeline=pipe)

pdf_paths = "/content/"

loader = PyPDFDirectoryLoader(
    path= pdf_paths,
    glob="*.pdf"
)
documents=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                chunk_overlap=100)

splits = text_splitter.split_documents(documents)

vectorstore_db = FAISS.from_documents(splits, embeddings) # create vector db for similarity search

# performs a similarity check and returns the top K embeddings
# that are similar to the question’s embeddings
retriever = vectorstore_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

custom_prompt_template = """welcome to rohanbot.
Context= {context}
History = {history}
Question= {question}
Helpful Answer:
"""


prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=["question", "context", "history"])

qa_chain_with_memory = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                                   retriever = vectorstore_db.as_retriever(),
                                                   return_source_documents = False,
                                                   chain_type_kwargs = {"verbose": True,
                                                                        "prompt": prompt,
                                                                        "memory": ConversationBufferMemory(
                                                                            input_key="question",
                                                                            memory_key="history",
                                                                            return_messages=True)})


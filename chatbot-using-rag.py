import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# Problem Statement: You are given a knowledge document over which you need to build a chatbot using Retrieval Augmented Generation (RAG). The RAG module should be capable of receiving user question as input and will need to output an answer to it based on the knowledge provided in the document

# %% [markdown]
# Let's install required libraries:
# 
# 1. LangChain’s flexible abstractions and extensive toolkit enables developers to harness the power of LLMs.
# 2. Pinecone is a cloud-based vector database service that provides a managed and scalable platform for similarity search and vector indexing
# 3. openai provides with LLM
# 4. The "tiktoken" library is a tool developed by OpenAI that allows you to count the number of tokens in a text string without making an API call to OpenAI's servers

# %% [code] {"jupyter":{"outputs_hidden":false}}
pip install langchain

# %% [code] {"jupyter":{"outputs_hidden":false}}
!pip install pinecone-client

# %% [code] {"jupyter":{"outputs_hidden":false}}
!pip install openai

# %% [code] {"jupyter":{"outputs_hidden":false}}
!pip install tiktoken

# %% [code] {"jupyter":{"outputs_hidden":false}}
import langchain
import pinecone

# TextLoader will be used to load data from Pan_card_services.txt to the document 
from langchain.document_loaders import TextLoader

# CharacterTextSplitter will be used to chunkify the document
from langchain.text_splitter import CharacterTextSplitter

# OpenAIEmbeddings will be used to convert individual chunks and query to vectors
from langchain.embeddings.openai import OpenAIEmbeddings

#Pinecone serves as vector-database
from langchain.vectorstores import Pinecone

#OpenAI will provide with LLM to generate response to the query
from langchain import OpenAI

#The RetrievalQAChain is a chain that combines a Retriever and a QA chain. 
# It is used to retrieve documents from a Retriever and then use a QA chain to answer a question 
# based on the retrieved documents.
from langchain.chains import RetrievalQA

# %% [markdown]
# Let's load data from Pan_card_services.txt to the document

# %% [code] {"jupyter":{"outputs_hidden":false}}
loader = TextLoader("../input/knowledgedocuments/Pan_card_services.txt")
document = loader.load()
print(document)

# %% [markdown]
# data is succefully loaded into the document
# 
# Let's chunkify(split document into chunks) the document

# %% [code] {"jupyter":{"outputs_hidden":false}}
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=60)
texts = text_splitter.split_documents(document)
print(len(texts))

# %% [markdown]
# document is split into 100 chunks

# %% [code] {"jupyter":{"outputs_hidden":false}}
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
openAi_api_key = user_secrets.get_secret("openAiKey")
pinecone_api_key = user_secrets.get_secret("pineconeKey")

# set the env variable in order to use openAI module
os.environ["OPENAI_API_KEY"] = openAi_api_key

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

# %% [markdown]
# I will initialize OpenAIEmbeddings and create embeddings object to convert chunks into vectors

# %% [code] {"jupyter":{"outputs_hidden":false}}
embeddings = OpenAIEmbeddings(openai_api_key=openAi_api_key)

# %% [markdown]
# following will take the chunks, convert them into vectors using embeddings(Embedding model) object, and store the vectors into Pinecone

# %% [code] {"jupyter":{"outputs_hidden":false}}
docsearch = Pinecone.from_documents(texts, embeddings, index_name="chatbot-rag-embeddings-index")

# %% [markdown]
# following will:
# 1. take query in prompt, convert query to vector and place this vector in vector database
# 2. then top k vectors semantically similar to query vector will chosen
# 3. top k vectors will be converted back to respective chunks and stuffed into prompt
# 4. new prompt containing these k chunks and query will be fed to openAi LLM to generate response

# %% [code] {"jupyter":{"outputs_hidden":false}}
qa = RetrievalQA.from_chain_type(llm = OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# %% [markdown]
# Now, we can ask questions and get answers

# %% [code] {"jupyter":{"outputs_hidden":false}}
query = "What is the cost/fees of a PAN card?"
result = qa({"query":query})
print(result["result"])

# %% [code] {"jupyter":{"outputs_hidden":false}}
query = "Can I apply for a PAN card if I am a non-resident Indian (NRI)?"
result = qa({"query":query})
print(result["result"])

# %% [code] {"jupyter":{"outputs_hidden":false}}
query = "How to apply for PAN card?"
result = qa({"query":query})
print(result["result"])
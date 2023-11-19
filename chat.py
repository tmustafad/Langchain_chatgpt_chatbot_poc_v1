from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import weaviate
from langchain.vectorstores import Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

loader = DirectoryLoader('./pdfs', glob='**/*.pdf')
data = loader.load()

print('Data size '+ str(len(data)))

text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key='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')

# connect to weaviate cluster

auth_config = weaviate.AuthApiKey(api_key= 'D0YP0KAiCZLpzQxqwKlUwcXNH07uUcvC3twE')
weaviate_url = 'https://chatbot-poc-s5b9f3g9.weaviate.network'

client = weaviate.Client(url = weaviate_url,
                         additional_headers={"X-OpenAI-Api-Key":'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'},
                            auth_client_secret=auth_config,startup_period=10)

# define input structure
# client.schema.delete_all()
# client.schema.get()
# schema = {
#     "classes": [
#         {
#             "class": "Chatbot",
#             "description": "Documents for chatbot poc",
#             "vectorizer": "text2vec-openai",
#             "moduleConfig": {"text2vec-openai": {"model": "ada", "type": "text"}},
#             "properties": [
#                 {
#                     "dataType": ["text"],
#                     "description": "The content of the paragraph",
#                     "moduleConfig": {
#                         "text2vec-openai": {
#                             "skip": False,
#                             "vectorizePropertyName": False,
#                         }
#                     },
#                     "name": "content",
#                 },
#             ],
#         },
#     ]
# }

# client.schema.create(schema)

vectorstore = Weaviate(client, "Chatbot", "content", attributes=["source"])

# load text into the vectorstore
# text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
# texts, meta = list(zip(*text_meta_pair))
# vectorstore.add_texts(texts, meta)

query = "What is the company name that has given the offer? "

# retrieve text related to the query
docs = vectorstore.similarity_search(query, top_k=20)

# define chain
chain = load_qa_chain(
    OpenAI(openai_api_key = "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",temperature=0), 
    chain_type="stuff")

# create answer
print(chain.run(input_documents=docs, question=query))
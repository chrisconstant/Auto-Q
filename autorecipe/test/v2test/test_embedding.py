from genai import Client, Credentials
from genai.extensions.langchain import LangChainEmbeddingsInterface
from genai.text.embedding import TextEmbeddingParameters

creds = {
    "api_key": "pak-whBjdbU__x9iGseK-ZU2q0xbxrI3mwEwgKms9UDBtlg",
    "api_endpoint": "https://bam-api.res.ibm.com",
}

client = Client(credentials=Credentials(**creds))
embeddings = LangChainEmbeddingsInterface(
    client=client,
    model_id="sentence-transformers/all-minilm-l6-v2",
    parameters=TextEmbeddingParameters(truncate_input_tokens=True)
)


query_embedding = embeddings.embed_query("Hello world!")
print(query_embedding)

doc_embedding = embeddings.embed_documents(["First document", "Second document"])
print(doc_embedding)

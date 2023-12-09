import pinecone
llmkey='LLM api key'
def vec_db():
    return pinecone.init(api_key='Pine cone API Key',environment='gcp-starter')
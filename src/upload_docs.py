# Load json add document embeddings and upload to Azure Search index

import os  
import json  
from openai import AzureOpenAI
from dotenv import load_dotenv  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  

from azure.search.documents.indexes.models import (  
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,     
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SearchField,  
    SemanticSearch,
    VectorSearch,  
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearch,    
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch,  
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)  

print("Please wait while we upload documents to Azure Search index...")

# The directory containing our documents.
DATA_DIR = "data/json"

load_dotenv()  

service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") 
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") 
key = os.getenv("AZURE_SEARCH_KEY") 
model = "ada002" 
credential = AzureKeyCredential(key)

# Create a client
client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_KEY"),  
  api_version = "2023-12-01-preview",
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Generate Document Embeddings using OpenAI Ada 002
# Read the text-sample.json
with open(f'{DATA_DIR}/library_data.json', 'r', encoding='utf-8') as file:
    input_data = json.load(file)

# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text, model=model):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Generate embeddings for content and category
for item in input_data:
    title = item['title']
    content = item['content']    
    title_embeddings = generate_embeddings(title)
    content_embeddings = generate_embeddings(content)    
    item['titleVector'] = title_embeddings
    item['contentVector'] = content_embeddings    

# Output embeddings to docVectors.json file
with open(f"{DATA_DIR}/output/docVectors.json", "w") as f:
    json.dump(input_data, f)
    
# Create a search index
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="categories", type=SearchFieldDataType.String, filterable=True),        
        SearchableField(name="doctype", type=SearchFieldDataType.String),
        SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    ]

# Configure the vector search configuration  
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            name="myExhaustiveKnn",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
            parameters=ExhaustiveKnnParameters(
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        )
    ],
    profiles=[
        VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw"),
        VectorSearchProfile(name="myExhaustiveKnnProfile", algorithm_configuration_name="myExhaustiveKnn")
    ]
)

semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        keywords_fields=[SemanticField(field_name="categories")],
        content_fields=[SemanticField(field_name="content")]
    )
)

# Create the semantic settings with the configuration
semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
index_client.delete_index(index)
result = index_client.create_or_update_index(index)
print(f"Index '{result.name}' created...")

# # Upload some documents to the index
with open(f'{DATA_DIR}/output/docVectors.json', 'r') as file:      
    documents = json.load(file) 
     
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
result = search_client.upload_documents(documents)
print(f"Uploaded {len(documents)} documents. (non batched)") 

# Upload some documents to the index  (batch)
# with open(f'{DATA_DIR}/output/docVectors.json', 'r') as file:  
#     documents = json.load(file)  
  
# # Use SearchIndexingBufferedSender to upload the documents in batches optimized for indexing  
# with SearchIndexingBufferedSender(  
#     endpoint=service_endpoint,  
#     index_name=index_name,  
#     credential=credential,  
# ) as batch_client:  
#     # Add upload actions for all documents  
#     batch_client.upload_documents(documents=documents)  
# print(f"Uploaded {len(documents)} documents.")  
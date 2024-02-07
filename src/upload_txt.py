import os
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    HnswAlgorithmConfiguration,
    SemanticPrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticSearch,
    SemanticConfiguration,
    SemanticField,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
)

from dotenv import load_dotenv

# from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import math
import tiktoken

load_dotenv()

# The directory containing our documents.
DATA_DIR = "data/extracted"

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "docindex-txt"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


# Loads the documents and split them into chunks.
def load_and_split_documents() -> list[dict]:
    # Load our data.
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"loaded {len(docs)} documents")

    # Split our documents.
    splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print(f"split into {len(split_docs)} documents")

    # Convert documents to a list of dictionaries.
    final_docs = []
    for i, doc in enumerate(split_docs):
        doc_dict = {
            "id": str(i),
            "content": doc.page_content,
            "sourcefile": os.path.basename(doc.metadata["source"]),
        }
        final_docs.append(doc_dict)

    return final_docs


# Returns an Azure AI Search index with the given name.
def get_index(name: str) -> SearchIndex:
    # The fields we want to index. The "embedding" field is a vector field that will be used for vector search.
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="sourcefile", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            # Size of the vector created by the text-embedding-ada-002 model.
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    # The "content" field should be prioritized for semantic ranking.
    semantic_config = SemanticConfiguration(
        name="default-txt",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=None,
            keywords_fields=[],
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE
                ),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            ),
        ],
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index.
    index = SearchIndex(
        name=name,
        fields=fields,
        semantic_search=semantic_search,
        vector_search=vector_search,
    )

    return index


# Initializes an Azure AI Search index with our custom data, using vector
def initialize(search_index_client: SearchIndexClient):
    aoai_client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-07-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # Load our data.
    docs = load_and_split_documents()

    # count the tokens in each document (for rag retrieval)
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_sizes = [len(encoding.encode(doc["content"])) for doc in docs]
    batch_size = 16
    num_batches = math.ceil(len(docs) / batch_size)

    # Embed our documents.
    print(
        f"embedding {len(docs)} documents in {num_batches} batches of {batch_size}. using embedding deployment {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}"
    )
    print(
        f"Total tokens: {sum(token_sizes)}, average tokens: {int(sum(token_sizes) / len(token_sizes))}"
    )
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(docs))
        batch_docs = docs[start_idx:end_idx]
        embeddings = aoai_client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=[doc["content"] for doc in batch_docs],
        ).data

        for j, doc in enumerate(batch_docs):
            doc["embedding"] = embeddings[j].embedding

    # Create an Azure Cognitive Search index.
    print(f"creating index {AZURE_SEARCH_INDEX_NAME}")
    index = get_index(AZURE_SEARCH_INDEX_NAME)
    search_index_client.create_or_update_index(index)

    # Upload our data to the index.
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    print(f"uploading {len(docs)} documents to index {AZURE_SEARCH_INDEX_NAME}")
    search_client.upload_documents(docs)


# Deletes existing Azure AI Search index.
def delete(search_index_client: SearchIndexClient):
    print(f"deleting index {AZURE_SEARCH_INDEX_NAME}")
    search_index_client.delete_index(AZURE_SEARCH_INDEX_NAME)


def main():
    try:
        search_index_client = SearchIndexClient(
            AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        delete(search_index_client)
        initialize(search_index_client)
        print("Upload completed")
    except Exception as ex:
        print("Upload failed")
        print(ex)


if __name__ == "__main__":
    main()

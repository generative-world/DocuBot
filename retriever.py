import weaviate
from config import WEAVIATE_URL
from chunking import chunk_document, embed_chunks

# Connect to the Weaviate instance
client = weaviate.Client(WEAVIATE_URL)

def store_document_in_db(text):
    """
    Breaks the document into chunks, generates embeddings, and stores them in Weaviate.
    """
    chunks = chunk_document(text)
    embeddings = embed_chunks(chunks)

    # Store each chunk with its embedding into Weaviate
    for i, chunk in enumerate(chunks):
        client.data_object.create(
            data_object={
                "text": chunk,
                "embedding": embeddings[i].tolist()  # Ensure embedding is stored as list
            },
            class_name="DocumentChunk"
        )

def retrieve_similar_documents(query, top_k=3):
    """
    Retrieves the top_k most similar document chunks from the vector database.
    """
    query_embedding = embed_chunks([query])[0]  # Embed the query
    result = client.query.get("DocumentChunk", ["text"])\
        .with_near_vector({"vector": query_embedding.tolist()})\
        .with_limit(top_k)\
        .do()

    return result['data']['Get']['DocumentChunk']

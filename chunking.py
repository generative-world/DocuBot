from sentence_transformers import SentenceTransformer
import textwrap

# Load the pre-trained embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def chunk_document(text, chunk_size=500):
    """
    Splits the document into smaller chunks of the specified size (in characters).
    """
    return textwrap.wrap(text, chunk_size)

def embed_chunks(chunks):
    """
    Embeds the document chunks using a pre-trained model.
    """
    embeddings = embedding_model.encode(chunks)
    return embeddings

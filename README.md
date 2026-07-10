Multimodal PDF Search Engine  ~ Semantic search across PDF text and figures using multimodal CLIP embeddings and Streamlit.

Multimodal PDF Search Engine is a Python-based AI application that enables semantic search across technical PDF documents by retrieving both relevant text passages and extracted figures. It uses CLIP multimodal embeddings to represent text and images in a shared embedding space, allowing natural language queries to return the most semantically similar content instead of relying on keyword matching.

Features:

* PDF text extraction using PyMuPDF
* Automatic image extraction from PDF pages
* Fixed-size text chunking with overlap
* Multimodal embeddings using OpenAI CLIP
* Semantic search using cosine similarity
* Retrieval of both text and image results
* Interactive Streamlit-based user interface
* Metadata tracking (PDF name, page number, chunk information)

How to Use:

1. Clone the repository.
2. Install the required libraries.
   * pip install -r requirements.txt
3. Place PDF documents inside:
   * data/raw_pdfs/
4. Run the preprocessing pipeline to:
   * extract text
   * extract images
   * chunk text
   * generate embeddings
5. Launch the application.
   * streamlit run streamlit_app.py
6. Enter a natural language query (e.g., "multimodal search architecture"), and the application will display the most relevant text chunks and figures.

Built With:

* Python 3
* PyMuPDF
* Hugging Face Transformers
* OpenAI CLIP (ViT-B/32)
* PyTorch
* NumPy
* scikit-learn
* Pillow
* Streamlit

Design:

* Modular architecture with separate preprocessing and retrieval components
* Shared embedding space for text and images using CLIP
* Cosine similarity for semantic retrieval
* Metadata-based mapping from embeddings back to original PDF pages
* Interactive web interface for querying and result visualization
* Designed as a beginner-friendly introduction to multimodal Retrieval-Augmented Generation (RAG) concepts

Project Workflow:

PDF Documents
      │
      ▼
Text & Image Extraction
      │
      ▼
Text Chunking
      │
      ▼
Generate CLIP Embeddings
      │
      ▼
Store Embeddings & Metadata
      │
      ▼
User Query
      │
      ▼
Query Embedding
      │
      ▼
Cosine Similarity Search
      │
      ▼
Top-K Text & Image Results
      │
      ▼
Streamlit Interface

Future Improvements:

* FAISS vector database for faster retrieval
* Caption-aware image search
* OCR support for scanned PDFs
* Section-aware semantic chunking
* Support for multiple embedding models
* Hybrid keyword and vector search

License: This project is licensed under the MIT License.

Contributions: Feel free to fork the repository, open an issue, or submit a pull request with improvements or additional features.

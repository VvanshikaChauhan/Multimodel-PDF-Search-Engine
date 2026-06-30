import streamlit as st

from src.retrieval_pipeline import MultimodalRetriever

st.set_page_config(page_title="Multimodal PDF Search", layout="wide")

st.title("Multimodal PDF Search Engine")
st.write("Search research PDFs across text chunks and extracted images.")

st.success("Streamlit is working!")

st.write("Loading retriever...")

retriever = MultimodalRetriever()

st.success("Retriever loaded!")

with st.sidebar:
    st.header("Search Settings")
    top_k = st.slider("Number of results", 1, 10, 5)
    st.write("PDF content must already be processed before searching.")

query = st.text_input("Enter your search query", placeholder="Example: user interface search button")

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching PDF embeddings..."):
            results = retriever.search(query, top_k=top_k)

        st.subheader("Text Results")
        for result in results["text_results"]:
            with st.expander(f"Score {result['score']:.4f} | Page {result['page_number']} | {result['filename']}"):
                st.write(result["text"])

        st.subheader("Image Results")
        for result in results["image_results"]:
            st.write(f"Score: {result['score']:.4f}")
            st.write(result["image_path"])
            st.image(result["image_path"], use_container_width=True)
import argparse
import os
from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load environment variables
load_dotenv(dotenv_path="PDF_Reader/.env")

CHROMA_PATH = "PDF_Reader/chroma"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

PROMPT_TEMPLATE = """Answer the following question using only the information from the context.

Context:
{context}

Question: {question}
Answer:"""



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load Chroma DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(query_text, k=1)
    filtered_results = [(doc, score) for doc, score in results if score >= 0.0]

    if not filtered_results:
        print("❌ No relevant results found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\n📌 QUERY PROMPT SENT TO LLM:\n")
    print(prompt)

    try:
        # Use HuggingFace pipeline
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-xl",
            device_map="auto",
            model_kwargs={"torch_dtype": "float16"},
            token=HUGGINGFACEHUB_API_TOKEN,
            trust_remote_code=True,
        )

        response_text = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
        print(f"\n🧠 Response:\n{response_text}")
    except Exception as e:
        print(f"\n❌ Failed to get LLM response: {e} {type(e)}")
        return

    sources = [doc.metadata.get("source", None) for doc, _ in filtered_results]
    print(f"\n📚 Sources: {sources}")

if __name__ == "__main__":
    main()


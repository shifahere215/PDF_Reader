
# import argparse
# import os
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load environment variables
# load_dotenv(dotenv_path="PDF_Reader/.env")

# CHROMA_PATH = "PDF_Reader/chroma"
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# PROMPT_TEMPLATE = """### Instruction:
# Answer the following question using only the information from the context.

# ### Context:
# {context}

# ### Question:
# {question}

# ### Response:"""

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     # Load Chroma DB
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

#     results = db.similarity_search_with_relevance_scores(query_text, k=1)
#     filtered_results = [(doc, score) for doc, score in results if score >= 0.0]

#     if not filtered_results:
#         print("❌ No relevant results found.")
#         return

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     print("\n📌 QUERY PROMPT SENT TO LLM:\n")
#     print(prompt)

#     try:
#         model_name = "mistralai/Mistral-7B-Instruct-v0.1"

#         # HuggingFace auth setup
#         os.environ["HF_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

#         tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             token=HUGGINGFACEHUB_API_TOKEN,
#         )

#         inputs = tokenizer(prompt, return_tensors="pt").to("mps")  # or "cpu"
#         outputs = model.generate(**inputs, max_new_tokens=150)
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         print(f"\n🧠 Response:\n{response_text}")

#     except Exception as e:
#         print(f"\n❌ Failed to get LLM response: {e} {type(e)}")
#         return

#     sources = [doc.metadata.get("source", None) for doc, _ in filtered_results]
#     print(f"\n📚 Sources: {sources}")

# if __name__ == "__main__":
#     main()


import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv(dotenv_path="PDF_Reader/.env")

CHROMA_PATH = "PDF_Reader/chroma"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Number of top documents to fetch (extended context)
CONTEXT_CHUNKS = 3

# Chain-of-thought prompt
PROMPT_TEMPLATE = """### Instruction:
Use the context to answer the question. Think step-by-step and explain clearly.

### Context:
{context}

### Question:
{question}

### Response:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load Chroma DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search for multiple top documents
    results = db.similarity_search_with_relevance_scores(query_text, k=CONTEXT_CHUNKS)
    filtered_results = [(doc, score) for doc, score in results if score >= 0.0]

    if not filtered_results:
        print("❌ No relevant results found.")
        return

    # Combine context chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\n📌 QUERY PROMPT SENT TO LLM:\n")
    print(prompt)

    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACEHUB_API_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=HUGGINGFACEHUB_API_TOKEN,
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("mps")
        outputs = model.generate(**inputs, max_new_tokens=250)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n🧠 Response:\n{response_text}")

    except Exception as e:
        print(f"\n❌ Failed to get LLM response: {e} {type(e)}")
        return

    sources = [doc.metadata.get("source", None) for doc, _ in filtered_results]
    print(f"\n📚 Sources: {sources}")

if __name__ == "__main__":
    main()

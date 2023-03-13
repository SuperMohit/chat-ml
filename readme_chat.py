from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter

import pathlib
import subprocess
import pickle
import os


dirName = "gitrepo"

chain = load_qa_with_sources_chain(OpenAI(temperature=0.2))

# caches the github repo with latest commit to the local machine
# TODO - implement repo update 
def get_github_docs(repo_owner, repo_name):
        # if repo is already cloned just return
        if not os.path.exists(dirName):                       
            os.mkdir(dirName)
            subprocess.check_call(
                f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
                cwd=pathlib.Path(dirName),
                shell=True,
            )  
        repo_path = pathlib.Path(dirName)
        # we are only interested in markdown files for Q&A bot
        # skip everything else
        markdown_files = list(repo_path.glob("cdk/**/*.md"))
        for markdown_file in markdown_files:
                print(f"markdown_file : {markdown_file}")
                with open(markdown_file, "r") as f:
                    relative_path = markdown_file.relative_to(repo_path)
                    github_url = f"https://github.com/{repo_owner}/{repo_name}/tree/master/{relative_path}"
                    ## create in-memory Documents with source and the page content
                    yield Document(page_content=f.read(), metadata={"source": github_url})

# delete the pickle in the path to rebuild the embeddings
if not os.path.exists("search_index.open.ai.pickle"):
    sources = get_github_docs("mongodb", "mongodbatlas-cloudformation-resources")
    ## processing the large documents would be expensive.
    ## OpenAI has token limit of 4096, so we need consise context
    ## In order to do that split the documents into chunk
    ## Save it in vectorized format using FAISS-
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    with open("search_index.open.ai.pickle", "wb") as f: 
        search_index = FAISS.from_documents(source_chunks, embedding=OpenAIEmbeddings())
        pickle.dump(search_index, f)


def print_answer(question):
    with open("search_index.open.ai.pickle", "rb") as f:
        search_index = pickle.load(f)
    return (
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )
    

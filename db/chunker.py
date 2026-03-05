from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

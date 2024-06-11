import os
from typing import List
import fitz

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file(self.path)
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_pdf_file(self.path)
        else:
            raise ValueError(
                f"Provided path {self.path} is neither a valid directory, a .pdf file, nor a .txt file."
            )

    def load_pdf_file(self, file_path):
        print(f"Loading PDF file: {file_path}")
        document = fitz.open(file_path)
        text = ""
        for page in document:
            text += page.get_text()
        self.documents.append(text)
    
    def load_file(self, file_path):
        print(f"Loading text file: {file_path}")
        with open(file_path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    self.load_file(file_path)
                elif file.endswith(".pdf"):
                    self.load_pdf_file(file_path)

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    # List all files in the current directory to confirm the presence of AlmanackNaval.pdf
    print(os.listdir('.'))
    
    # Load the PDF file
    loader = TextFileLoader("AlmanackNaval.pdf")
    documents = loader.load_documents()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(documents)
    print(f"Number of chunks: {len(chunks)}")
    print(chunks[0][:1000])  # Print the first 1000 characters of the first chunk to verify
    print("--------")
    print(chunks[1][:1000])  # Print the first 1000 characters of the second chunk to verify
    print("--------")
    print(chunks[-2][:1000])  # Print the first 1000 characters of the second last chunk to verify
    print("--------")
    print(chunks[-1][:1000])  # Print the first 1000 characters of the last chunk to verify

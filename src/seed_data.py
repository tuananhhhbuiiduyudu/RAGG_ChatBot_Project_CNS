import os 
import json 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings , OllamaEmbeddings
from langchain.docstore.document import Document
from crawl import craw_web
from dotenv import load_dotenv

load_dotenv()

def load_data_from_local(filename : str , directory : str) -> tuple: 
    file_path = os.path.join(directory , filename)
    
    with open(file_path , 'r') as file : 
        data = json.load(file)
    
    print(f"Data loaded from {file_path}")
    
    return data , filename.rsplit('.' , 1)[0].replace('_' , ' ')

def seed_faiss(collection_name : str  , filename : str , directory : str , save_path: str = "./vectorstores") -> FAISS : 
 
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    
    # Đọc dữ liệu từ file_local : 
    local_data , doc_name = load_data_from_local(filename , directory)
    
    # Chuyên dữ liệu thành documents :
    documents = [
        Document(
            page_content=doc.get('page_content') or '',
            metadata={
                'source': doc['metadata'].get('source') or '',
                'content_type': doc['metadata'].get('content_type') or 'text/plain',
                'title': doc['metadata'].get('title') or '',
                'description': doc['metadata'].get('description') or '',
                'language': doc['metadata'].get('language') or 'en',
                'doc_name': doc_name,
                'start_index': doc['metadata'].get('start_index') or 0
            }
        )
        for doc in local_data
    ]
    vectorstore = FAISS.from_documents(documents , embedding= embeddings)
    
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(os.path.join(save_path, collection_name))

    print(f"FAISS vectorstore saved to {os.path.join(save_path, collection_name)}")
    return vectorstore

def seed_faiss_live(URL : str , collection_name : str , doc_name : str , save_path: str = "./vectorstores" ) -> FAISS :   

    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    
    documens = craw_web(URL)
    
    # Cập nhật metada cho mỗi documents với giá trị mặc định 
    for doc in documens : 
        metadata = {
            'source' : doc.metadata.get('source') or '',
            'content_type' : doc.metadata.get('content_type') or '',
            'title': doc.metadata.get('title') or '',
            'description': doc.metadata.get('description') or '',
            'language': doc.metadata.get('language') or 'en',
            'doc_name': doc_name,
            'start_index': doc.metadata.get('start_index') or 0
        }
        doc.metadata = metadata
    
    vectorstore = FAISS.from_documents(documens , embeddings)
    os.makedirs(save_path , exist_ok= True)
    vectorstore.save_local(os.path.join(save_path , collection_name))
    print(f"FAISS vectorstore saved to {os.path.join(save_path, collection_name)}")
    return vectorstore

def connect_to_faiss(collection_name : str ,  persist_path = "./vectorstores") -> FAISS : 

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        folder_path=f"{persist_path}/{collection_name}",
        embeddings = embeddings,
        allow_dangerous_deserialization=True  # thêm dòng này!
    )
    print(f"Loaded FAISS vectorstore from {persist_path}/{collection_name}")
    return vectorstore
    
    
def main():
#     # Test seed_faiss với dũ liệu local
#     seed_faiss(
#     collection_name= 'data_test'
#     filename='stack.json',
#     directory='data',
#     use_ollama=False
# )       
    # Test seed_faiss_live với dữ liệu trực tiếp 
    seed_faiss_live(
        URL='https://www.stack-ai.com/docs',
        collection_name= "data_test_live",
        doc_name='stack-ai',
    )
    

# Chạy main() nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main()
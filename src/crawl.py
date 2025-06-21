import os 
import json 
from langchain_community.document_loaders import WebBaseLoader , RecursiveUrlLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import re 
from dotenv import load_dotenv

load_dotenv()

def bs4_extractor(html : str) -> str : 
    soup = BeautifulSoup(html , "html.parser")
    return re.sub(r"\n\n+" , "\n\n" , soup.text).strip()

def craw_web(url_data) :
    loader = RecursiveUrlLoader(url= url_data , extractor= bs4_extractor , max_depth= 4)
    docs = loader.load()
    
    print("length"  , len(docs))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000 , chunk_overlap = 500)
    all_splitter = text_splitter.split_documents(docs)
    
    print("length_all_splits :" , len(all_splitter))
    
    return all_splitter 

def web_base_loader(url_data):
    loader = WebBaseLoader(web_path=url_data)
    docs = loader.load()
    
    print("length:" , len(docs))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000 , chunk_overlap = 500)
    all_splitter = text_splitter.split_documents(docs)
    
    print("length all_spliiter :" , len(all_splitter))
    
    return all_splitter

def save_data_local(documents , filename , directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory , filename)
    
    data_to_save = [{'page_content' : doc.page_content , 'metadata' : doc.metadata} for doc in documents]
    
    with open(file_path , 'w') as file :
        json.dump(data_to_save , file , indent = 4)
    
    print(f"Data saved to {file_path}")
    
## Test thử các hàm đã tạo 

def main():
    data = craw_web('https://www.stack-ai.com/docs')
    
    save_data_local(data ,  "stack.json" , 'data')
    print('data' , data)

if __name__ == "__main__":
    main()
    
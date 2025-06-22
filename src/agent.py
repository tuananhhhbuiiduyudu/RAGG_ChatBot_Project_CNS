from langchain.tools.retriever import create_retriever_tool 
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent , AgentType
from langchain.agents import AgentExecutor , create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder 
from seed_data import seed_faiss , connect_to_faiss 
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os 

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key : 
    raise ValueError("GROQ_API_KEY not found in environment variables")
# llm = ChatGroq(groq_api_key = groq_api_key , model_name = "Llama3-8b-8192")

def get_retriever(collection_name : str = "data_test" , persist_path = "./vectorstores") -> EnsembleRetriever : 
    ## Kết nối tới FAISS và tạo vector retrierver
    try : 
        vectorstore = connect_to_faiss(collection_name=collection_name , persist_path= persist_path)
        faiss_retrierver = vectorstore.as_retriever(
            search_type = "similaryty",
            search_kwargs = {"k" : 4}
        )
        
        # Tạo BM25 từ toàn bộ documents 
        documents = [
            Document(page_content = doc.page_content , metadata = doc.metadata)
            for doc in vectorstore.similarity_search("" , k = 100)
        ]
        
        if not documents : 
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
        
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4
        
        # Kết hợp cả 2 lại 
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retrierver , bm25_retriever],
            weights=[0.7 , 0.3]
        )
        
        return ensemble_retriever
    
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)
    
tool = create_retriever_tool(
    get_retriever(),
    "find",
    "Search for information of Stack AI."
)

def get_llm_and_agent(_retriever, model_choice="groq"):
    if model_choice == "groq":
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="Llama3-8b-8192",
            temperature=0.0,
            streaming= True 
        )
    else:
        raise ValueError("Model không được hỗ trợ hiện tại")
    
    tools = [tool]
    
    system = """You are an expert at AI. Your name is UDU_GOLOBAL."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    # Khởi tạo agent
    
    agent = create_openai_functions_agent(llm = llm  , tools = tools  , prompt = prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Khởi tạo retriever và agent
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)

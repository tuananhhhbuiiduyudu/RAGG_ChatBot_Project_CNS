"""
File chính để chạy ứng dụng Chatbot AI
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời
"""
# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv  # Đọc file .env chứa API key
from seed_data import seed_milvus, seed_milvus_live  # Hàm xử lý dữ liệu
from agent import get_retriever as get_openai_retriever, get_llm_and_agent as get_openai_agent
from local_ollama import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="AI Assistant",  # Tiêu đề tab trình duyệt
        page_icon="💬",  # Icon tab
        layout="wide"  # Giao diện rộng
    )

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    """
    load_dotenv()  # Đọc API key từ file .env
    setup_page()  # Thiết lập giao diện

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Phần 1: Chọn Embeddings Model
        st.header("🔤 Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["OpenAI", "Ollama"]
        )
        use_ollama_embeddings = (embeddings_choice == "Ollama")
        
        # Phần 2: Cấu hình Data
        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"]
        )
        
        # Xử lý nguồn dữ liệu dựa trên embeddings đã chọn
        if data_source == "File Local":
            handle_local_file(use_ollama_embeddings)
        else:
            handle_url_input(use_ollama_embeddings)
            
        # Thêm phần chọn collection để query
        st.header("🔍 Collection để truy vấn")
        collection_to_query = st.text_input(
            "Nhập tên collection cần truy vấn:",
            "data_test",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )
        
        # Phần 3: Chọn Model để trả lời
        st.header("🤖 Model AI")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["OpenAI GPT-4", "OpenAI Grok", "Ollama (Local)"]
        )
        
        return model_choice, collection_to_query
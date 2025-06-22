import streamlit as st
from dotenv import load_dotenv
from crawl import craw_web, save_data_local
from seed_data import seed_faiss, seed_faiss_live
from agent import get_retriever, get_llm_and_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def setup_page():
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="💬",
        layout="wide"
    )


def initialize_app():
    load_dotenv()
    setup_page()


def setup_sidebar():
    with st.sidebar:
        st.title("⚙️ Cấu hình")

        st.header("🔤 Embeddings Model")
        st.write("Sử dụng HuggingFace Embeddings (all-MiniLM-L6-v2)")

        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"],
            help="Chọn cách tải dữ liệu vào vectorstore"
        )

        if data_source == "File Local":
            handle_local_file()
        else:
            handle_url_input()

        st.header("🔍 Collection để truy vấn")
        collection_to_query = st.text_input(
            "Nhập tên collection cần truy vấn:",
            "data_test",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )

        st.header("🤖 Model AI")
        st.write("Sử dụng Grok (Llama3-8b-8192)")
        model_choice = "groq"

        return model_choice, collection_to_query


def handle_local_file():
    collection_name = st.text_input("Tên collection trong FAISS:", "data_test")
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")

    if st.button("Tải dữ liệu từ file"):
        if not collection_name or not filename or not directory:
            st.error("Vui lòng nhập đầy đủ thông tin!")
            return

        with st.spinner("Đang tải dữ liệu..."):
            try:
                vectorstore = seed_faiss(
                    collection_name=collection_name,
                    filename=filename,
                    directory=directory,
                    save_path="./vectorstores"
                )
                st.success(f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")
                st.info(f"Số documents: {len(vectorstore.docstore._dict)}")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")


def handle_url_input():
    collection_name = st.text_input("Tên collection trong FAISS:", "data_test_live")
    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    doc_name = st.text_input("Tên tài liệu:", "stack-ai")

    if st.button("Crawl dữ liệu"):
        if not collection_name or not url or not doc_name:
            st.error("Vui lòng nhập đầy đủ thông tin!")
            return

        with st.spinner("Đang crawl dữ liệu..."):
            try:
                vectorstore = seed_faiss_live(
                    URL=url,
                    collection_name=collection_name,
                    doc_name=doc_name,
                    save_path="./vectorstores"
                )
                st.success(f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
                st.info(f"Số documents: {len(vectorstore.docstore._dict)}")
            except Exception as e:
                st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")


def setup_chat_interface(model_choice):
    st.title("💬 AI Assistant")
    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và xAI Grok")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs


def handle_user_input(msgs, agent_executor):
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Stack AI!"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]
            try:
                response = agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": chat_history
                    },
                    {"callbacks": [st_callback]}
                )
                output = response["output"]
            except Exception as e:
                output = f"Lỗi khi xử lý yêu cầu: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)


def main():
    initialize_app()
    model_choice, collection_to_query = setup_sidebar()
    msgs = setup_chat_interface(model_choice)

    try:
        retriever = get_retriever(collection_name=collection_to_query)
        agent_executor = get_llm_and_agent(retriever, model_choice="groq")
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo agent: {str(e)}. Hãy chắc chắn rằng bạn đã tải dữ liệu vào collection này.")
        return

    handle_user_input(msgs, agent_executor)


if __name__ == "__main__":
    main()

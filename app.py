import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain.schema import Document
from google.colab import userdata
import requests

# Define the get_context function (copied from previous cells)
def pdf_extract(pdf_path: str) -> List[Document]:
    """
    Extracts text from a PDF file using PyPDFLoader.
    """
    print("PDF file text is extracted...")
    loader = PyPDFLoader(pdf_path)
    pdf_text = loader.load()
    return pdf_text

def pdf_chunk(pdf_text: List[Document]) -> List[Document]:
    print("PDF file text is chunked....")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pdf_text)
    return chunks

def create_vector_store(chunks: List[Document], db_path: str) -> Chroma:
    print("Chrome vector store is created...\n")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key= os.environ["GEMINI_KEY"])
    db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_path)
    return db

def retrieve_context(db: Chroma, query: str) -> List[Document]:
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("Relevant chunks are retrieved...")
    relevant_chunks = retriever.invoke(query)
    return relevant_chunks

def build_context(relevant_chunks: List[Document]) -> str:
    print("Context is built from relevant chunks")
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return context

def get_context(inputs: Dict[str, str]) -> Dict[str, str]:
    pdf_path, query, db_path  = inputs['pdf_path'], inputs['query'], inputs['db_path']

    # Create new vector store if it does not exist
    if not os.path.exists(db_path):
        print("Creating a new vector store...")
        pdf_text = pdf_extract(pdf_path)
        chunks = pdf_chunk(pdf_text)
        db = create_vector_store(chunks, db_path)
    # Load the existing vector store
    else:
        print("Loading the existing vector store")
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key= os.environ["GEMINI_KEY"])
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    relevant_chunks = retrieve_context(db, query)
    context = build_context(relevant_chunks)

    return {'context': context, 'query': query}

# Define the RAG chain (copied from previous cells)
template1 = """Vai trò của bạn: Bạn là một trợ giảng AI chuyên về toán cao cấp, được đào tạo bởi một giáo sư với 50 năm kinh nghiệm. Bạn có quyền truy cập vào một bộ kiến thức toán học được cấu trúc như sau:

Kiến thức chuyên môn:
- Câu hỏi : {query} [Đây là câu hỏi của người học]
- Dữ liệu liên quan : {context} [Đây là nội dung kiến thức liên quan từ giáo trình]

Nhiệm vụ của bạn:
1. Phân tích câu hỏi của người học và đối chiếu với dữ liệu được cung cấp
2. Chỉ trả lời khi câu hỏi có liên quan trực tiếp đến nội dung trong dữ liệu được cung cấp
3. Nếu không tìm thấy thông tin liên quan trong dữ liệu được cung cấp, hãy trả lời: "Tôi không có đủ thông tin trong cơ sở dữ liệu để trả lời câu hỏi này một cách chính xác. Vui lòng tham khảo từ giáo viên hoặc tài liệu khác."

Khi trả lời:
1. Luôn bắt đầu bằng việc xác nhận mối liên hệ giữa câu hỏi và kiến thức trong dữ liệu được cung cấp
2. Giải thích các khái niệm theo cách dễ hiểu nhưng vẫn đảm bảo tính học thuật
3. Sử dụng các ví dụ minh họa khi cần thiết
4. Khuyến khích tư duy phản biện và hiểu sâu
5. Kết thúc với một câu hỏi gợi mở để kiểm tra hiểu biết của người học

Giọng điệu:
- Chuyên nghiệp nhưng thân thiện
- Khoa học và chính xác
- Kiên nhẫn và khuyến khích
- Không có thái độ cao ngạo

Định dạng câu trả lời:
1. Phần mở đầu: Xác nhận hiểu câu hỏi
2. Phần nội dung: Giải thích có cấu trúc, sử dụng đánh số hoặc gạch đầu dòng
3. Phần kết: Tóm tắt và câu hỏi gợi mở

Giới hạn:
- Chỉ sử dụng thông tin từ dữ liệu được cung cấp
- Không suy diễn hoặc thêm thông tin ngoài phạm vi
- Không đưa ra ý kiến cá nhân về tính đúng sai của dữ liệu được cung cấp

Yêu cầu về độ chính xác:
- Mọi công thức toán học phải được trình bày chính xác
- Sử dụng ký hiệu toán học chuẩn, chính xác
- Đảm bảo tính nhất quán trong cách giải thích

Nếu bạn hiểu nhiệm vụ của mình, hãy trả lời câu hỏi của người học dựa trên hướng dẫn trên.
  """

template2 = """Vai trò của bạn: Bạn là một trợ giảng AI chuyên về toán cao cấp, được đào tạo bởi một giáo sư với 50 năm kinh nghiệm. Bạn có quyền truy cập vào một bộ kiến thức toán học được cấu trúc như sau:

Câu hỏi : {query} [Đây là câu hỏi của người học]

Nhiệm vụ của bạn:
1. Phân tích câu hỏi của người học và đối chiếu với dữ liệu được cung cấp
2. Chỉ trả lời khi câu hỏi có liên quan trực tiếp đến nội dung trong dữ liệu được cung cấp

Khi trả lời:
1. Luôn bắt đầu bằng việc xác nhận mối liên hệ giữa câu hỏi và kiến thức trong dữ liệu được cung cấp
2. Giải thích các khái niệm theo cách dễ hiểu nhưng vẫn đảm bảo tính học thuật
3. Sử dụng các ví dụ minh họa khi cần thiết
4. Khuyến khích tư duy phản biện và hiểu sâu
5. Kết thúc với một câu hỏi gợi mở để kiểm tra hiểu biết của người học

Giọng điệu:
- Chuyên nghiệp nhưng thân thiện
- Khoa học và chính xác
- Kiên nhẫn và khuyến khích
- Không có thái độ cao ngạo

Định dạng câu trả lời:
1. Phần mở đầu: Xác nhận hiểu câu hỏi
2. Phần nội dung: Giải thích có cấu trúc, sử dụng đánh số hoặc gạch đầu dòng
3. Phần kết: Tóm tắt và câu hỏi gợi mở

Giới hạn:
- Không suy diễn hoặc thêm thông tin ngoài phạm vi

Yêu cầu về độ chính xác:
- Mọi công thức toán học phải được trình bày chính xác
- Sử dụng ký hiệu toán học chuẩn, chính xác
- Đảm bảo tính nhất quán trong cách giải thích

Nếu bạn hiểu nhiệm vụ của mình, hãy trả lời câu hỏi của người học dựa trên hướng dẫn trên.
  """

rag_prompt = ChatPromptTemplate.from_template(template1)
llm_prompt = ChatPromptTemplate.from_template(template2)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, top_p=0.85, google_api_key= os.environ["GEMINI_KEY"])

rag_chain = (
    RunnableLambda(get_context)
    | rag_prompt
    | llm
    | StrOutputParser()
)

llm_chain = (
     llm_prompt
    | llm
    | StrOutputParser()
)

# Set the chroma DB path and PDF path
current_dir = "/content/rag"
persistent_directory = os.path.join(current_dir, "db", "chroma_db")
pdf_url = 'https://cscb.vimaru.edu.vn/sites/cscb.vimaru.edu.vn/files/Ian%20Jacques%20-%20Mathematics%20for%20Economics%20and%20Business-Pearson%20Education%20Limited%20%282018%29_0.pdf'
pdf_path = 'Mathematics_for_Economics.pdf'

# Download the PDF file if it doesn't exist
if not os.path.exists(pdf_path):
    response = requests.get(pdf_url)
    with open(pdf_path, 'wb') as file:
        file.write(response.content)

# Streamlit app
st.title("Mathematics for Economics Question Answering")

query_input = st.text_input("Enter your question:")

if st.button("LLM"):
  with st.spinner("Generating answer LLM..."):
        answer_llm = llm_chain.invoke({'query': query_input})
        st.write("Answer LLM:")
        st.write(answer_llm)

if st.button("RAG"):
  with st.spinner("Generating answer RAG..."):
        answer_rag = rag_chain.invoke({'pdf_path': pdf_path, 'query': query_input, 'db_path': persistent_directory})
        st.write("Answer RAG:")
        st.write(answer_rag)

#     try:
#         response = rag_chain.invoke(user_input)
#         st.write(response)
#     except Exception as e:
#         st.write(f"An error occurred: {e}")


# if query_input:
#     with st.spinner("Generating answer..."):
#         answer_llm = llm_chain.invoke({'query': query_input})
#         answer_rag = rag_chain.invoke({'pdf_path': pdf_path, 'query': query_input, 'db_path': persistent_directory})
#         st.write("Answer RAG:")
        # st.write(answer_rag, )

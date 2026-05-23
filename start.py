## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

## 환경변수 불러오기
from dotenv import load_dotenv,dotenv_values
load_dotenv()



############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장
# pdf를 chunking, parsing 등 작업하기 위해서는 이 프로그램이 실행되는 컴퓨터에 파일을 저장해야 작업 가능.
def save_uploadedfile(uploadedfile: UploadedFile) -> str : 
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read()) 
    return file_path

## 2: 저장된 PDF 파일을 Document로 변환
# pdf 파일을 읽고 Document 형태로 변환하는 과정.
# Document는 langchain에서 텍스트와 메타데이터를 함께 저장하는 객체. (page_content, metadata)
def pdf_to_documents(pdf_path:str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents

## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장
# 문서 덩어리가 벡터로 전환되는 임베딩 과정을 OpenAI 모델로 수행
# FAISS는 벡터를 저장하고 검색하는 라이브러리. 여기서는 FAISS를 사용해서 벡터DB를 구축하고, 로컬에 저장.
# faiss_index 폴더가 생성되고, 그 안에 벡터DB가 저장됨. 이후 RAG 처리할 때 이 벡터DB를 불러와서 관련 문서를 검색하는 데 사용.
def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")



############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):


    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    ## 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    ## 사용자 질문을 기반으로 관련문서 3개 검색 
    ## invoke는 retriever라는 함수에서 input으로 user_question을 넣어서 검색을 수행하는 메서드.
    retrieve_docs : List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()
    ## 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs


## get_rag_chain 함수는 질문, 질문 관련 문서(context)를 입력으로 받아서 프롬프트가 더 풍부해지고,
## 모델이 그 프롬프트를 기반으로 응답을 생성하는 RAG 체인을 반환하는 함수
def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")

    return custom_rag_prompt | model | StrOutputParser()



############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################

@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []
    
    # 이미지 저장용 폴더 생성
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):  #  각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72이 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
        pix.save(image_path)  # PNG 형태로 저장
        image_paths.append(image_path)  # 경로를 저장
        
    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def main():
    st.set_page_config(page_title="청약 FAQ 챗봇", layout="wide")
    left_column, right_column = st.columns([1, 1])
    
    with left_column:   
        st.header('청약 FAQ 챗봇')
        pdf_doc = st.file_uploader('PDF Uploader', type='pdf')
        button = st.button('PDF 업로드하기')
        if pdf_doc and button:
            
            # (1단계) PDF 문서를 벡터DB에 저장
            with st.spinner('PDF 문서 저장 중...'):
                st.text('여기까지 구현됨')
                pdf_path = save_uploadedfile(pdf_doc) # PDF 파일을 임시폴더에 저장
                pdf_document = pdf_to_documents(pdf_path) # PDF 파일을 Document 형태로 변환
                smaller_documents = chunk_documents(pdf_document) # Document를 더 작은 document로 변환
                save_to_vector_store(smaller_documents) # Document를 벡터DB에 저장

            # (3단계) PDF를 이미지로 변환해서 세션 상태로 임시 저장
            with st.spinner('PDF 문서를 이미지로 변환 중...'):
                images = convert_pdf_to_images(pdf_path)
                st.session_state.images = images
                
        user_question = st.text_input("PDF 문서에 대해 질문을 입력하세요",
                                        placeholder="예시) 무순위 청약 시에도 부부 중복신청이 가능한가요?")
        
        if user_question:
            response, context = process_question(user_question)
            st.text(response)
            for document in context:
                with st.expander('관련 문서'):
                    st.text(document.page_content)
                    file_path = document.metadata.get('source', '')
                    page_number = document.metadata.get('page', 0)+1
                    button_key = f'link_{file_path}_{page_number}'
                    reference_button = st.button(f'{os.path.basename(file_path)} pg.{page_number}', key=button_key)
                    if reference_button:
                        st.session_state.page_number = str(page_number)
        
        with right_column:
            # page_number 호출
            page_number = st.session_state.get('page_number')
            if page_number:
                page_number = int(page_number)
                image_folder = "PDF_이미지"
                images = sorted(os.listdir(image_folder), key=natural_sort_key)
                print(images)
                images_paths = [os.path.join(image_folder, img) for img in images]
                print(page_number)
                print(images_paths[page_number-1])
                display_pdf_page(images_paths[page_number-1], page_number)
                
if __name__ == "__main__":
    main()
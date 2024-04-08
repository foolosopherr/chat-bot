import os
import tempfile
import streamlit as st
# from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")


@st.cache_resource()
def configure_retriever(local_folder="./data/"):
   """Configures a retriever using documents from a local folder."""

   docs = []
   folder_path = local_folder
   print(os.listdir(folder_path))
   for filename in os.listdir(folder_path):
       filepath = os.path.join(folder_path, filename)
       loader = TextLoader(filepath)
       docs.extend(loader.load())

   # Split documents
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
   splits = text_splitter.split_documents(docs)

   # Create embeddings and store in vectordb
   model_kwargs = {'device': 'cpu'}
   model_name = 'sentence-transformers/all-MiniLM-L6-v2'
   encode_kwargs = {'normalize_embeddings': False}

   embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
   vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

   # Define retriever
   retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

   return retriever



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()

hf_api_key = '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'



# uploaded_files = st.sidebar.file_uploader(
#     label="Upload TXT files", type=["txt"], accept_multiple_files=True
# )

# if not uploaded_files:
#     st.info("Please upload TXT documents to continue.")
#     st.stop()

retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
# )
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm, retriever=retriever, memory=memory, verbose=True
# )

# создаем шаблон для промта
prompt_template = """Используй контекст для ответа на вопрос, пользуясь следующими правилами:

Не изменяй текст, который находится в кавычках.
В конце обязательно добавь ссылку на полный документ
{answer}
url: {url}
"""

llm = HuggingFaceHub(repo_id='IlyaGusev/fred_t5_ru_turbo_alpaca',
                    huggingfacehub_api_token=hf_api_key,
                    model_kwargs={'temperature':0, 'max_length':256},
                    prompt=PROMPT,
                    )


qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)



PROMPT = PromptTemplate(
template=prompt_template, input_variables=['answer', 'url']
)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_chroma import Chroma  # Only import once after the fix

load_dotenv()
import streamlit as st

st.set_page_config(
    page_title="🦙💬 Llama 2 Chatbot",
    page_icon="🦙",
    layout="wide"
)

st.header("Welcome to Llama 2 Chatbot! 🦙💬")
st.write("Chat with a powerful AI model based on Llama 2.")


# Rest of your application code...

# Inject meta tags using unsafe_allow_html
meta_tags = """
<head>
<title>Meta Tags — Preview, Edit and Generate</title>
<meta name="title" content="Meta Tags — Preview, Edit and Generate" />
<meta name="description" content="With Meta Tags you can edit and experiment with your content then preview how your webpage will look on Google, Facebook, Twitter and more!" />

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website" />
<meta property="og:url" content="https://llama2.streamlit.app/" />
<meta property="og:title" content="Meta Tags — Preview, Edit and Generate" />
<meta property="og:description" content="With Meta Tags you can edit and experiment with your content then preview how your webpage will look on Google, Facebook, Twitter and more!" />
<meta property="og:image" content="https://metatags.io/images/meta-tags.png" />

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image" />
<meta property="twitter:url" content="https://llama2.streamlit.app/" />
<meta property="twitter:title" content="Meta Tags — Preview, Edit and Generate" />
<meta property="twitter:description" content="With Meta Tags you can edit and experiment with your content then preview how your webpage will look on Google, Facebook, Twitter and more!" />
<meta property="twitter:image" content="https://metatags.io/images/meta-tags.png" />
</head>
"""

# Inject the meta tags into the Streamlit app
st.markdown(meta_tags, unsafe_allow_html=True)

# Streamlit UI starts below
st.title("Welcome to My Streamlit App")
st.write("This app has custom meta tags for social media preview.")




# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores (question, answer) pairs

# Streamlit UI
st.title("Best RAG MODEL 🚀")

# 📌 **List of PDFs to Load** (Add file names here)
pdf_files = ["resume.pdf",  "ticket.pdf"]  # Add more PDFs as needed

# Load and combine text from multiple PDFs
all_docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    all_docs.extend(loader.load())  # Append all documents from each PDF

# Split text into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

# Create vector store (in-memory mode)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory=None,  # Ensures it's in-memory
    client_settings=Settings(is_persistent=False)  # Disable persistence
)

# Create retriever (using MMR for diversity)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=100, timeout=None)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

# Define system prompt to allow general knowledge responses
system_prompt = (
    "You are Vishnu AI assistant that answers questions using both retrieved context and general knowledge. "
    "If relevant context is available, use it to provide an accurate response. "
    "If the context does not contain the answer, rely on your general knowledge. "
    "Keep responses clear and concise, limiting them to two or three sentences.\n\n"
    "{context}"
)

# Define chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Streamlit chat input
query = st.chat_input("Ask me anything...")

if query:
    # Create RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Get response from RAG
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]

    # If no relevant retrieved context, allow LLM to answer freely
    if not response["context"]:
        answer = llm.invoke(query)  # General knowledge response

    # Store chat history (latest at the top, keep only last 5)
    st.session_state.chat_history.insert(0, (query, answer))
    st.session_state.chat_history = st.session_state.chat_history[:5]

# Display chat history
st.subheader("Chat History")
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")
    st.write("---")

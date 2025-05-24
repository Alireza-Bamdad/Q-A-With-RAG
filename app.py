import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import streamlit as st
import requests
import os

# Configure Streamlit page
st.set_page_config(
    page_title=" مشاوره انتخاب رشته",
    page_icon="🎓",
    layout="wide"
)
st.markdown("""
<style>
    body, [class*="css"]  {
        direction: rtl;
        text-align: rigrtlht;
        font-family: 'B Nazanin', 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions to avoid reloading data and model
@st.cache_data
def load_data():
    with open("advisor_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item["input"] for item in data]
    answers = [item["output"] for item in data]
    
    return questions, answers, data

@st.cache_resource
def load_model():
    model_name = "HooshvareLab/bert-fa-zwnj-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    return tokenizer, model, device

@st.cache_data
def create_embeddings_and_index(_questions):
    question_embeddings = create_embeddings(_questions)
    
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(question_embeddings).astype('float32'))
    
    return question_embeddings, index

# Create embeddings for questions using a Persian language model
def create_embeddings(texts):
    tokenizer, model, device = load_model()

    if isinstance(texts, str):
        texts = [texts]

    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    token_embeddings = model_output[0]
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.cpu().numpy()

def retrieve_relevant_docs(query, questions, answers, index, k=5):
    """Retrieve relevant documents for RAG"""
    query_embedding = create_embeddings(query)
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)

    relevant_docs = []
    for i, idx in enumerate(indices[0]):
        if idx < len(questions) and scores[0][i] > 0.75:  # آستانه برای RAG
            relevant_docs.append({
                "question": questions[idx],
                "answer": answers[idx],
                "score": float(scores[0][i])
            })

    return relevant_docs

def call_groq_api(prompt, api_key):
    """Call Groq API for text generation"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system", 
                "content": "شما یک مشاور تحصیلی هستید که به زبان فارسی پاسخ می‌دهید. بر اساس اطلاعات ارائه شده، پاسخی مفید و دقیق و با لهن دوستانه صحبت کنید ارائه دهید. اگر اطلاعات کافی ندارید، صادقانه بگویید."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"خطا در اتصال به Groq API: {str(e)}"
    except Exception as e:
        return f"خطا در پردازش پاسخ: {str(e)}"

def generate_rag_response(query, relevant_docs, api_key):
    """Generate response using RAG approach"""
    if not relevant_docs:
        no_context_prompt = f"""
        سوال کاربر: {query}

        هیچ اطلاعات مرتبطی در پایگاه داده مشاوره تحصیلی پیدا نشد. 
        اگر این سوال مرتبط با مشاوره تحصیلی، انتخاب رشته یا راهنمایی آموزشی است، 
        لطفاً بر اساس دانش عمومی خود پاسخ مختصری ارائه دهید.
        اگر سوال کاملاً نامرتبط است، مودبانه توضیح دهید که فقط در زمینه مشاوره تحصیلی کمک می‌کنید.
        """
        return call_groq_api(no_context_prompt, api_key)
    
    # Create context from relevant documents
    context = "اطلاعات مرتبط از پایگاه داده:\n\n"
    for i, doc in enumerate(relevant_docs[:3]):  
        context += f"سوال {i+1}: {doc['question']}\n"
        context += f"پاسخ {i+1}: {doc['answer']}\n"
        context += f"امتیاز مرتبط بودن: {doc['score']:.3f}\n\n"
    
    prompt = f"""
{context}

سوال جدید کاربر: {query}

بر اساس اطلاعات بالا، لطفاً پاسخی جامع و مفید به سوال کاربر ارائه دهید.
- از اطلاعات موجود استفاده کنید
- اگر لازم است، اطلاعات را ترکیب کنید
- پاسخ را به زبان فارسی و به صورت واضح , و دوستانه  ارائه دهید
- اگر اطلاعات کافی نیست، صادقانه بگویید
-در نظر داشته باش که پاسخ تولید شده مناسب دانش اموز پایه نهم باشد
-سوالات درمورد انتخاب رشته پایه نهم در ایران است 

پاسخ:
"""
    
    return call_groq_api(prompt, api_key)

# Main Streamlit App
def main():
    st.title("🎓سیستم مشاوره تحصیلی پایه نهم")
    st.markdown("هر سوالی درمورد انتخاب رشته داری بپرس")
    st.markdown("---")
    

    api_key = GORK_API_KEY
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        with st.spinner("در حال بارگذاری داده‌ها و مدل..."):
            # Load data
            questions, answers, data = load_data()
            st.session_state.questions = questions
            st.session_state.answers = answers
            st.session_state.data = data
            
            # Load model
            load_model()
            
            # Create embeddings and index
            question_embeddings, index = create_embeddings_and_index(questions)
            st.session_state.question_embeddings = question_embeddings
            st.session_state.index = index
            st.session_state.initialized = True
            


    # Chat interface
    st.subheader("💬 سوال خود را بپرسید")
    
    # User input
    user_query = st.text_area(
        "سوال:",
        placeholder="مثال: من به ریاضی علاقه دارم و از حفظ کردن بدم میاد، چه رشته‌ای پیشنهاد می‌دید؟",
        height=100,
        key="user_input"
    )
    
    # Submit button
    if st.button("🤖 پاسخ مشاور", type="primary") or user_query:
        if user_query.strip():
            with st.spinner("در حال جستجو و تولید پاسخ..."):
                # Retrieve relevant documents
                relevant_docs = retrieve_relevant_docs(
                    user_query, 
                    st.session_state.questions, 
                    st.session_state.answers, 
                    st.session_state.index
                )
                
                # Generate RAG response
                rag_response = generate_rag_response(user_query, relevant_docs, api_key)
                
                # Display response
                st.subheader("🤖 پاسخ هوش مصنوعی:")
                st.success(rag_response)
                
                # Display retrieved context in expander
                with st.expander("📚 اطلاعات بازیابی شده (منابع)"):
                    if relevant_docs:
                        st.write(f"**تعداد منابع یافت شده:** {len(relevant_docs)}")
                        for i, doc in enumerate(relevant_docs):
                            with st.container():
                                st.markdown(f"**منبع {i+1} (امتیاز: {doc['score']:.3f})**")
                                st.markdown(f"**سوال:** {doc['question']}")
                                st.markdown(f"**پاسخ:** {doc['answer']}")
                                st.markdown("---")
                    else:
                        st.write("هیچ منبع مرتبطی یافت نشد. پاسخ بر اساس دانش عمومی مدل تولید شده است.")
        else:
            st.warning("لطفاً سوال خود را وارد کنید.")

 
if __name__ == "__main__":
    main()
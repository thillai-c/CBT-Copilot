import streamlit as st
from transformers import pipeline
import torch
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="CBT Copilot - Your Therapeutic Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333333 !important;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0 !important;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #7b1fa2 !important;
    }
    .chat-message strong {
        color: inherit !important;
        font-weight: 600;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333333 !important;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404 !important;
    }
    /* Fix for Streamlit's default text colors */
    .stMarkdown p {
        color: inherit !important;
    }
    /* Ensure proper contrast in all text elements */
    div[data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the CBT-Copilot model with caching"""
    try:
        # Show detailed loading status
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("🔄 Initializing model loading...")
        progress_bar.progress(20)
        
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        
        progress_text.text(f"🔄 Loading model on {device_info}... (This may take 2-15 minutes on first run)")
        progress_bar.progress(40)
        
        pipe = pipeline(
            "text-generation", 
            model="thillaic/CBT-Copilot",
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        progress_text.text("✅ Model loaded successfully!")
        progress_bar.progress(100)
        
        # Clear progress indicators after a moment
        time.sleep(1)
        progress_text.empty()
        progress_bar.empty()
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_response(pipe, messages, max_length=512, temperature=0.7, top_p=0.9):
    """Generate response from the model"""
    try:
        with st.spinner("Thinking..."):
            response = pipe(
                messages,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')
                # Remove the input prompt from the response
                if isinstance(generated_text, list):
                    # If it's a conversation format, get the last assistant message
                    for msg in reversed(generated_text):
                        if msg.get('role') == 'assistant':
                            return msg.get('content', 'I apologize, but I had trouble generating a response.')
                elif isinstance(generated_text, str):
                    return generated_text
            
            return "I apologize, but I had trouble generating a response. Please try again."
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I'm having technical difficulties. Please try again."

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 CBT Copilot</h1>
        <p>Your Personal Cognitive Behavioral Therapy Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Settings")
        
        # Model parameters
        st.markdown("#### Generation Parameters")
        max_length = st.slider("Max Response Length", 100, 1000, 512, 50)
        temperature = st.slider("Temperature (Creativity)", 0.1, 1.0, 0.7, 0.1)
        top_p = st.slider("Top-p (Diversity)", 0.1, 1.0, 0.9, 0.1)
        
        st.markdown("---")
        
        # Information section
        st.markdown("""
        <div class="sidebar-content">
            <h4>ℹ️ About CBT-Copilot</h4>
            <p>This AI assistant is designed to support cognitive behavioral therapy practices with empathetic, safe, and privacy-preserving conversations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Important Disclaimer</strong><br>
            This tool is for educational and support purposes only. It is not a replacement for professional mental health care. Please consult with qualified healthcare providers for serious mental health concerns.
        </div>
        """, unsafe_allow_html=True)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    
    # Load model
    if not st.session_state.model_loaded:
        pipe = load_model()
        if pipe is not None:
            st.session_state.pipe = pipe
            st.session_state.model_loaded = True
            st.success("✅ CBT-Copilot model loaded successfully!")
        else:
            st.error("❌ Failed to load the model. Please check your internet connection and try again.")
            st.stop()
    
    # Display conversation history
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>CBT Copilot:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Share what's on your mind... I'm here to listen and support you.")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Prepare messages for the model
        model_messages = st.session_state.messages.copy()
        
        # Generate response
        if st.session_state.model_loaded:
            response = generate_response(
                st.session_state.pipe, 
                model_messages,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to display the new messages
            st.rerun()
    
    # Welcome message for new users
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="chat-message assistant-message">
            <strong>CBT Copilot:</strong><br>
            Hello! I'm here to support you with cognitive behavioral therapy techniques. 
            I can help you explore your thoughts, feelings, and behaviors in a safe and confidential space. 
            What would you like to talk about today?
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>🔒 Your conversations are private and not stored permanently. 
        This session will reset when you refresh the page.</p>
        <p>Built with ❤️ using Streamlit and Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
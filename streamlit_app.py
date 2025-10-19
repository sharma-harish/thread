"""
Streamlit Chat UI
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import our multi-agent system
try:
    from langgraph_project.main import get_or_create_graph, compile_graph
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Weave Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c, #d62728);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .flow-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .user-flow { background-color: #e8f5e8; border-left-color: #2e7d32; }
    .documentation-flow { background-color: #fff3e0; border-left-color: #f57c00; }
    .general-flow { background-color: #fce4ec; border-left-color: #c2185b; }
    .more-info-flow { background-color: #e1f5fe; border-left-color: #0277bd; }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        background-color: #fafafa;
    }
    .message-bubble {
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #007bff;
        color: white;
        margin-left: auto;
    }
    .assistant-bubble {
        background-color: #e9ecef;
        color: #333;
        margin-right: auto;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-loading { background-color: #ffc107; animation: pulse 1s infinite; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "system_status" not in st.session_state:
        st.session_state.system_status = "offline"

def initialize_graph():
    """Initialize the multi-agent graph with enhanced error handling."""
    if st.session_state.graph is None:
        with st.spinner("Initializing weave multi-agent system..."):
            try:
                st.session_state.graph = get_or_create_graph()
                st.session_state.system_status = "online"
                st.success("✅ Multi-agent system initialized successfully!")
                return True
            except Exception as e:
                st.session_state.system_status = "offline"
                st.error(f"❌ Failed to initialize multi-agent system: {e}")
                return False
    return True

async def process_query_async(query: str) -> Dict[str, Any]:
    """Enhanced async query processing with detailed metrics."""
    if not st.session_state.graph:
        return {"error": "Graph not initialized"}
    
    try:
        start_time = time.time()
        
        # Prepare the input state
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "message": {"type": "user", "logic": ""}
        }
        
        # Execute the graph
        result = await st.session_state.graph.ainvoke(initial_state)
        
        processing_time = time.time() - start_time
        
        # Extract response and flow information
        response = ""
        flow_type = "unknown"
        
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                response = last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                response = last_message['content']
        
        if "message" in result:
            flow_type = result["message"].get("type", "unknown")
        
        return {
            "response": response,
            "flow_type": flow_type,
            "processing_time": processing_time,
            "full_result": result
        }
    except Exception as e:
        return {"error": str(e)}

def process_query(query: str) -> Dict[str, Any]:
    """Synchronous wrapper for async query processing."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, process_query_async(query))
                return future.result()
        else:
            return loop.run_until_complete(process_query_async(query))
    except RuntimeError:
        return asyncio.run(process_query_async(query))

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Enhanced message display with better styling."""
    if is_user:
        st.markdown(f"""
        <div class="message-bubble user-bubble">
            <strong>You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        flow_type = message.get('flow_type', 'unknown')
        # confidence = message.get('confidence', 0.0)
        
        # Flow-specific styling
        flow_colors = {
            "user": "#2e7d32",
            "documentation": "#f57c00", 
            "general": "#c2185b",
            "more-info": "#0277bd"
        }
        flow_icons = {
            "user": "👤",
            "documentation": "📚",
            "general": "❓",
            "more-info": "❔"
        }
        
        color = flow_colors.get(flow_type, "#666")
        icon = flow_icons.get(flow_type, "🤖")
        
        st.markdown(f"""
        <div class="message-bubble assistant-bubble">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="color: {color}; font-weight: bold;">
                    {icon} {flow_type.title()} Flow
                </span>
            </div>
            <strong>Assistant:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main advanced Streamlit application."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Thread - Advanced Multi-Agent System</h1>', unsafe_allow_html=True)
    initialize_graph()
    # Sidebar
    with st.sidebar:
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    display_chat_interface()

def display_chat_interface():
    """Display the chat interface."""
    st.subheader("💬 Chat Interface")
    
    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message, message.get("role") == "user")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        if not st.session_state.graph:
            st.error("Please initialize the multi-agent system first.")
            return
        
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Process query
        with st.spinner("🤔 Processing..."):
            result = process_query(prompt)
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # Add assistant response
            assistant_message = {
                "role": "assistant",
                "content": result.get("response", "No response generated"),
                "flow_type": result.get("flow_type", "unknown"),
                "processing_time": result.get("processing_time", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            st.rerun()

if __name__ == "__main__":
    main()
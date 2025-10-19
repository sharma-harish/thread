"""
Streamlit Chat UI for Multi-Agent Workflow

This application provides a user-friendly chat interface for the Thread multi-agent system.
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import our multi-agent system
try:
    from langgraph_project.main import get_or_create_graph, compile_graph
    from langgraph_project.evaluation.evaluator import run_quick_evaluation
    from langgraph_project.evaluation.evaluation_runner import EvaluationRunner
    from langgraph_project.evaluation.multi_agent_evaluator import FlowType
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Thread - Multi-Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #7b1fa2;
    }
    .flow-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .user-flow { background-color: #e8f5e8; color: #2e7d32; }
    .documentation-flow { background-color: #fff3e0; color: #f57c00; }
    .general-flow { background-color: #fce4ec; color: #c2185b; }
    .more-info-flow { background-color: #e1f5fe; color: #0277bd; }
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

def get_flow_color(flow_type: str) -> tuple:
    """Get color scheme for flow type."""
    colors = {
        "user": ("user-flow", "👤"),
        "documentation": ("documentation-flow", "📚"),
        "general": ("general-flow", "❓"),
        "more-info": ("more-info-flow", "❔")
    }
    return colors.get(flow_type, ("", "🤖"))

def initialize_graph():
    """Initialize the multi-agent graph."""
    if st.session_state.graph is None:
        with st.spinner("Initializing multi-agent system..."):
            try:
                st.session_state.graph = get_or_create_graph()
                st.success("✅ Multi-agent system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize multi-agent system: {e}")
                return False
    return True

async def process_query_async(query: str) -> Dict[str, Any]:
    """Process query using the multi-agent system."""
    if not st.session_state.graph:
        return {"error": "Graph not initialized"}
    
    try:
        # Prepare the input state
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "message": {"type": "user", "logic": ""}
        }
        
        # Execute the graph
        result = await st.session_state.graph.ainvoke(initial_state)
        
        # Extract response
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
            "full_result": result
        }
    except Exception as e:
        return {"error": str(e)}

def process_query(query: str) -> Dict[str, Any]:
    """Synchronous wrapper for async query processing."""
    try:
        # Try to get the existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a running event loop, we need to use asyncio.run_coroutine_threadsafe
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, process_query_async(query))
                return future.result()
        else:
            return loop.run_until_complete(process_query_async(query))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(process_query_async(query))

def display_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling."""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        flow_type = message.get('flow_type', 'unknown')
        flow_class, flow_icon = get_flow_color(flow_type)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="flow-indicator {flow_class}">
                {flow_icon} {flow_type.title()} Flow
            </div>
            <strong>Assistant:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)

def run_evaluation():
    """Run evaluation and display results."""
    if not st.session_state.graph:
        st.error("Please initialize the multi-agent system first.")
        return
    
    with st.spinner("Running evaluation..."):
        try:
            # Run quick evaluation
            results = asyncio.run(run_quick_evaluation())
            
            # Store results in session state
            st.session_state.evaluation_results = results
            
            st.success("✅ Evaluation completed!")
            return results
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            return None

def display_evaluation_results():
    """Display evaluation results."""
    if not st.session_state.evaluation_results:
        st.info("No evaluation results available. Run an evaluation first.")
        return
    
    results = st.session_state.evaluation_results
    
    st.subheader("📊 Evaluation Results")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Test Cases", len(results))
    
    successful_cases = len([r for r in results if not r.error_message])
    with col2:
        st.metric("Successful Cases", successful_cases)
    
    success_rate = successful_cases / len(results) if results else 0
    with col3:
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"Test Case {i}: {result.test_case.question[:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Question:**", result.test_case.question)
                st.write("**Expected Flow:**", result.test_case.expected_flow.value)
                st.write("**Actual Flow:**", result.actual_flow.value if result.actual_flow else "None")
            
            with col2:
                st.write("**Flow Accuracy:**", f"{result.flow_accuracy:.3f}" if result.flow_accuracy else "N/A")
                st.write("**Response Quality:**", f"{result.response_quality:.3f}" if result.response_quality else "N/A")
                st.write("**Execution Time:**", f"{result.execution_time:.3f}s" if result.execution_time else "N/A")
            
            if result.actual_response:
                st.write("**Response:**", result.actual_response[:200] + "..." if len(result.actual_response) > 200 else result.actual_response)
            
            if result.error_message:
                st.error(f"**Error:** {result.error_message}")

# Main application
def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Thread - Multi-Agent Chat System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize system
        if st.button("🚀 Initialize Multi-Agent System", type="primary"):
            if initialize_graph():
                st.session_state.messages = []
                st.session_state.conversation_history = []
        
        st.divider()
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.graph:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System Not Initialized")
        
        st.divider()
        
        # Evaluation section
        st.subheader("🧪 Evaluation")
        if st.button("🔍 Run Quick Evaluation"):
            run_evaluation()
        
        if st.button("📊 Show Evaluation Results"):
            display_evaluation_results()
        
        st.divider()
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
        
        # Flow type selector for testing
        st.subheader("🎯 Test Flow Types")
        flow_type = st.selectbox(
            "Select flow type for testing:",
            ["user", "documentation", "general", "more-info"]
        )
        
        st.info(f"Selected: {flow_type} flow")
    
    # Main chat interface
    st.subheader("💬 Chat with the Multi-Agent System")
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message, message.get("role") == "user")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about users, documentation, or general questions..."):
        if not st.session_state.graph:
            st.error("Please initialize the multi-agent system first using the sidebar.")
            return
        
        # Add user message to chat
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        display_message(user_message, is_user=True)
        
        # Process query
        with st.spinner("🤔 Processing your query..."):
            start_time = time.time()
            result = process_query(prompt)
            processing_time = time.time() - start_time
        
        if "error" in result:
            st.error(f"❌ Error processing query: {result['error']}")
        else:
            # Add assistant response to chat
            assistant_message = {
                "role": "assistant",
                "content": result.get("response", "No response generated"),
                "flow_type": result.get("flow_type", "unknown"),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            display_message(assistant_message)
            
            # Show processing time
            st.caption(f"⏱️ Processed in {processing_time:.2f}s")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Built with ❤️ using Streamlit, LangGraph, and LangChain
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

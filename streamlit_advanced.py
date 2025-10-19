"""
Advanced Streamlit Chat UI with Enhanced Features

This application provides an advanced chat interface with evaluation dashboard,
flow visualization, and comprehensive system monitoring.
"""

import streamlit as st
import asyncio
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
    from langgraph_project.evaluation.evaluator import run_comprehensive_evaluation
    from langgraph_project.evaluation.evaluation_runner import EvaluationRunner
    from langgraph_project.evaluation.multi_agent_evaluator import FlowType
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
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = []
    if "flow_statistics" not in st.session_state:
        st.session_state.flow_statistics = {
            "user": 0, "documentation": 0, "general": 0, "more-info": 0
        }
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
        confidence = 0.0
        
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                response = last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                response = last_message['content']
        
        if "message" in result:
            flow_type = result["message"].get("type", "unknown")
            confidence = result["message"].get("confidence", 0.0)
        
        # Update flow statistics
        if flow_type in st.session_state.flow_statistics:
            st.session_state.flow_statistics[flow_type] += 1
        
        # Store performance metrics
        performance_metric = {
            "timestamp": datetime.now(),
            "query": query,
            "flow_type": flow_type,
            "processing_time": processing_time,
            "confidence": confidence,
            "response_length": len(response)
        }
        st.session_state.performance_metrics.append(performance_metric)
        
        return {
            "response": response,
            "flow_type": flow_type,
            "confidence": confidence,
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
        confidence = message.get('confidence', 0.0)
        
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
                <span style="margin-left: auto; font-size: 0.8rem; color: #666;">
                    Confidence: {confidence:.1%}
                </span>
            </div>
            <strong>Assistant:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)

def display_dashboard():
    """Display comprehensive dashboard with metrics and visualizations."""
    st.subheader("📊 System Dashboard")
    
    # System status and metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if st.session_state.system_status == "online" else "🔴"
        st.metric("System Status", f"{status_color} {st.session_state.system_status.title()}")
    
    with col2:
        total_queries = len(st.session_state.performance_metrics)
        st.metric("Total Queries", total_queries)
    
    with col3:
        avg_processing_time = (
            sum(m["processing_time"] for m in st.session_state.performance_metrics) / 
            len(st.session_state.performance_metrics)
        ) if st.session_state.performance_metrics else 0
        st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
    
    with col4:
        total_messages = len(st.session_state.messages)
        st.metric("Chat Messages", total_messages)
    
    # Flow statistics visualization
    if st.session_state.performance_metrics:
        st.subheader("📈 Flow Distribution")
        
        flow_counts = st.session_state.flow_statistics
        flow_df = pd.DataFrame(list(flow_counts.items()), columns=['Flow Type', 'Count'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(flow_df, values='Count', names='Flow Type', 
                        title="Query Flow Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(flow_df, use_container_width=True)
        
        # Performance over time
        st.subheader("⏱️ Performance Over Time")
        
        perf_df = pd.DataFrame(st.session_state.performance_metrics)
        perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
        
        fig = px.line(perf_df, x='timestamp', y='processing_time', 
                     title="Processing Time Trend")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent queries table
        st.subheader("📋 Recent Queries")
        recent_df = perf_df.tail(10)[['timestamp', 'query', 'flow_type', 'processing_time']]
        st.dataframe(recent_df, use_container_width=True)

def run_comprehensive_evaluation():
    """Run comprehensive evaluation with progress tracking."""
    if not st.session_state.graph:
        st.error("Please initialize the multi-agent system first.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Starting comprehensive evaluation...")
        progress_bar.progress(10)
        
        # Create evaluation runner
        runner = EvaluationRunner()
        
        status_text.text("Loading test cases...")
        progress_bar.progress(25)
        
        # Run evaluation
        status_text.text("Running evaluation tests...")
        progress_bar.progress(50)
        
        results = asyncio.run(runner.run_comprehensive_evaluation())
        
        progress_bar.progress(90)
        status_text.text("Generating report...")
        
        st.session_state.evaluation_results = results
        
        progress_bar.progress(100)
        status_text.text("✅ Evaluation completed!")
        
        return results
    except Exception as e:
        st.error(f"❌ Evaluation failed: {e}")
        return None

def display_evaluation_dashboard():
    """Display comprehensive evaluation dashboard."""
    if not st.session_state.evaluation_results:
        st.info("No evaluation results available. Run an evaluation first.")
        return
    
    results = st.session_state.evaluation_results
    overall_summary = results['overall_summary']
    
    st.subheader("🧪 Evaluation Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Test Cases", overall_summary['total_test_cases'])
    
    with col2:
        st.metric("Successful Cases", overall_summary['total_successful_cases'])
    
    with col3:
        success_rate = overall_summary['overall_success_rate']
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col4:
        avg_exec_time = overall_summary.get('average_execution_time', 0)
        st.metric("Avg Execution Time", f"{avg_exec_time:.3f}s")
    
    # Flow breakdown
    st.subheader("📊 Flow Performance Breakdown")
    
    flow_breakdown = overall_summary['flow_breakdown']
    breakdown_df = pd.DataFrame([
        {
            'Flow': flow_name,
            'Total Cases': data['total_cases'],
            'Successful Cases': data['successful_cases'],
            'Success Rate': data['success_rate']
        }
        for flow_name, data in flow_breakdown.items()
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(breakdown_df, x='Flow', y='Success Rate',
                    title="Success Rate by Flow Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(breakdown_df, use_container_width=True)
    
    # Overall scores
    if 'overall_scores' in overall_summary:
        st.subheader("📈 Overall Scores")
        
        scores_df = pd.DataFrame([
            {'Metric': metric, 'Score': score}
            for metric, score in overall_summary['overall_scores'].items()
        ])
        
        fig = px.bar(scores_df, x='Metric', y='Score',
                    title="Overall Evaluation Scores")
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main advanced Streamlit application."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Thread - Advanced Multi-Agent Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # System initialization
        if st.button("🚀 Initialize System", type="primary"):
            initialize_graph()
        
        # System status
        st.subheader("📊 System Status")
        status_indicator = "🟢" if st.session_state.system_status == "online" else "🔴"
        st.write(f"{status_indicator} Status: {st.session_state.system_status.title()}")
        
        st.divider()
        
        # Navigation
        st.subheader("🧭 Navigation")
        page = st.selectbox(
            "Select Page:",
            ["💬 Chat Interface", "📊 Dashboard", "🧪 Evaluation", "⚙️ Settings"]
        )
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📊 Reset Metrics"):
            st.session_state.performance_metrics = []
            st.session_state.flow_statistics = {
                "user": 0, "documentation": 0, "general": 0, "more-info": 0
            }
            st.rerun()
    
    # Main content based on selected page
    if page == "💬 Chat Interface":
        display_chat_interface()
    elif page == "📊 Dashboard":
        display_dashboard()
    elif page == "🧪 Evaluation":
        display_evaluation_page()
    elif page == "⚙️ Settings":
        display_settings_page()

def display_chat_interface():
    """Display the chat interface."""
    st.subheader("💬 Advanced Chat Interface")
    
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
                "confidence": result.get("confidence", 0.0),
                "processing_time": result.get("processing_time", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            st.rerun()

def display_evaluation_page():
    """Display the evaluation page."""
    st.subheader("🧪 Evaluation & Testing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Run Evaluation")
        
        if st.button("🚀 Run Comprehensive Evaluation", type="primary"):
            run_comprehensive_evaluation()
        
        if st.button("⚡ Quick Test"):
            # Run a quick test with sample queries
            test_queries = [
                "What is Carol Johnson's contact number?",
                "How do I set up LLM call traces?",
                "What's the weather like today?",
                "Tell me about Carol's account"
            ]
            
            with st.spinner("Running quick test..."):
                for query in test_queries:
                    result = process_query(query)
                    st.write(f"**{query}** → {result.get('flow_type', 'unknown')} flow")
    
    with col2:
        if st.session_state.evaluation_results:
            display_evaluation_dashboard()

def display_settings_page():
    """Display the settings page."""
    st.subheader("⚙️ System Settings")
    
    # Configuration options
    st.subheader("🔧 Configuration")
    
    # Model settings
    st.write("**Model Configuration**")
    model_type = st.selectbox("Primary Model", ["OpenAI", "Google"])
    
    # System behavior
    st.write("**System Behavior**")
    auto_initialize = st.checkbox("Auto-initialize system on startup", value=True)
    show_confidence = st.checkbox("Show confidence scores", value=True)
    enable_tracing = st.checkbox("Enable Weave tracing", value=True)
    
    # Chat settings
    st.write("**Chat Settings**")
    max_messages = st.slider("Max chat messages", 10, 100, 50)
    auto_scroll = st.checkbox("Auto-scroll to latest message", value=True)
    
    # Save settings
    if st.button("💾 Save Settings"):
        st.success("Settings saved!")

if __name__ == "__main__":
    main()

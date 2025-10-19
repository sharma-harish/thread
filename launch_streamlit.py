#!/usr/bin/env python3
"""
Streamlit Launcher for Thread Multi-Agent System

This script provides an easy way to launch different Streamlit applications.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_streamlit_app(app_file: str, port: int = 8501):
    """Run a Streamlit application."""
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            app_file, "--server.port", str(port)
        ]
        print(f"🚀 Launching Streamlit app: {app_file}")
        print(f"📡 Server will be available at: http://localhost:{port}")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit app: {e}")

def main():
    """Main launcher function."""
    print("🤖 Thread Multi-Agent System - Streamlit Launcher")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Please install it first:")
        print("   pip install streamlit")
        return
    
    # Available apps
    app = {
            "name": "Advanced Dashboard",
            "file": "streamlit_app.py",
            "description": "Advanced dashboard with analytics and evaluation"
        }
    
    while True:
        app_path = Path(app["file"])

        if not app_path.exists():
            print(f"❌ App file not found: {app_path}")
            continue

        # Ask for port
        port_input = input(f"Enter port number (default 8501): ").strip()
        try:
            port = int(port_input) if port_input else 8501
        except ValueError:
            port = 8501
            print(f"Invalid port, using default: {port}")

        print(f"\n🚀 Launching {app['name']}...")
        run_streamlit_app(app["file"], port)

if __name__ == "__main__":
    main()
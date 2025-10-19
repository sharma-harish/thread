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
    apps = {
        "1": {
            "name": "Basic Chat Interface",
            "file": "streamlit_app.py",
            "description": "Simple chat interface with basic features"
        },
        "2": {
            "name": "Advanced Dashboard",
            "file": "streamlit_advanced.py", 
            "description": "Advanced dashboard with analytics and evaluation"
        }
    }
    
    print("\n📱 Available Applications:")
    for key, app in apps.items():
        print(f"  {key}. {app['name']}")
        print(f"     {app['description']}")
    
    print("\n🎯 Choose an application to launch:")
    
    while True:
        choice = input("\nEnter your choice (1-2) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        if choice in apps:
            app = apps[choice]
            app_path = Path(app['file'])
            
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
            run_streamlit_app(app['file'], port)
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, or 'q'.")

if __name__ == "__main__":
    main()

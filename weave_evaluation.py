"""
Simple Weave Leaderboard Example

An example showing how to create leaderboards using Weave.
"""

import asyncio
import sys
from pathlib import Path
# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from langgraph_project.evaluation.weave_evaluator import (
    run_simple_weave_evaluation
)


async def demo_evaluation():
    """Demo Weave evaluation."""
    print("🚀 Simple Weave Evaluation Demo")
    print("="*50)
    
    try:
        evaluation = await run_simple_weave_evaluation()
        print(f"✅ Evaluation completed successfully!")
        print(f"📊 Check your Weave dashboard for detailed results")
        return evaluation
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

async def main():
    """Main demo function."""
    print("🤖 Weave Evaluation Demo")
    print("="*60)
    
    print("This demo shows how to integrate Weave leaderboards with your evaluation system:")
    print("• Simple Weave evaluations with custom scorers")

    try:
        result = await demo_evaluation()
        results = [("Simple Weave Evaluation", result)]
        print(f"\n🚀 Running demos...")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
    
    print(f"\n{'='*60}")
    print("DEMO SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results:
        status = "✅ Success" if result else "❌ Failed"
        print(f"{name}: {status}")
    
    print(f"\n📊 Check your Weave dashboard for all results and leaderboards")
    print(f"🔍 Use the individual functions in your own scripts")


if __name__ == "__main__":
    asyncio.run(main())

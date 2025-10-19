"""
Simple Weave Leaderboard Example

An example showing how to create leaderboards using Weave.
"""

import asyncio

from src.langgraph_project.evaluation.weave_evaluator import (
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


# async def demo_leaderboard_creation():
#     """Demo leaderboard creation with multiple evaluations."""
#     print("\n🏆 Leaderboard Creation Demo")
#     print("="*50)
#
#     try:
#         leaderboard_ref = await run_leaderboard_demo()
#         print(f"✅ Leaderboard created successfully!")
#         print(f"📊 Check your Weave dashboard for the leaderboard")
#         return leaderboard_ref
#     except Exception as e:
#         print(f"❌ Leaderboard creation failed: {e}")
#         return None


# async def demo_custom_evaluations():
#     """Demo creating custom evaluations for different scenarios."""
#     import asyncio
#     print("\n🛠️  Custom Evaluations Demo")
#     print("="*50)
#
#     # Load test cases
#     test_cases = load_test_cases("src/langgraph_project/evaluation/datasets/comprehensive_test_cases.json")
#
#     # Create different evaluation scenarios
#     evaluations = []
#
#     # Scenario 1: All flows mixed
#     print("📊 Running Scenario 1: AllFlows_Mixed")
#     all_flows_cases = test_cases[:10]  # First 10 cases
#     eval1 = await run_weave_evaluation(all_flows_cases, "AllFlows_Mixed")
#     evaluations.append(eval1)
#     print(f"✅ Created 'AllFlows_Mixed' evaluation")
#
#     # Scenario 2: User flow only
#     user_cases = [tc for tc in test_cases if tc.expected_flow == FlowType.USER][:5]
#     if user_cases:
#         print("📊 Running Scenario 2: UserFlow_Only")
#         eval2 = await run_weave_evaluation(user_cases, "UserFlow_Only")
#         evaluations.append(eval2)
#         print(f"✅ Created 'UserFlow_Only' evaluation")
#
#     # Scenario 3: Documentation flow only
#     doc_cases = [tc for tc in test_cases if tc.expected_flow == FlowType.DOCUMENTATION][:5]
#     if doc_cases:
#         print("📊 Running Scenario 3: DocumentationFlow_Only")
#         eval3 = await run_weave_evaluation(doc_cases, "DocumentationFlow_Only")
#         evaluations.append(eval3)
#         print(f"✅ Created 'DocumentationFlow_Only' evaluation")
#
#     # Create leaderboard from all evaluations
#     if evaluations:
#         print("🏆 Creating custom leaderboard...")
#         leaderboard_ref = await create_leaderboard(evaluations, "Custom Multi-Agent Leaderboard")
#         print(f"✅ Created custom leaderboard with {len(evaluations)} evaluations")
#         print(f"📊 Check your Weave dashboard for the leaderboard")
#         return leaderboard_ref
#
#     return None


# async def demo_flow_comparison():
#     """Demo comparing different flow types."""
#     import asyncio
#     print("\n📊 Flow Comparison Demo")
#     print("="*50)
#
#     # Load test cases
#     test_cases = load_test_cases("src/langgraph_project/evaluation/datasets/comprehensive_test_cases.json")
#
#     # Create evaluations for each flow type
#     flow_evaluations = []
#     flow_types = [FlowType.USER, FlowType.DOCUMENTATION, FlowType.GENERAL, FlowType.MORE_INFO]
#
#     for i, flow_type in enumerate(flow_types, 1):
#         flow_cases = [tc for tc in test_cases if tc.expected_flow == flow_type][:3]
#         if flow_cases:
#             eval_name = f"{flow_type.value.upper()}_Flow"
#             print(f"📊 Running Flow Evaluation {i}: {eval_name}")
#             evaluation = await run_weave_evaluation(flow_cases, eval_name)
#             flow_evaluations.append(evaluation)
#             print(f"✅ Created {eval_name} evaluation")
#
#
#     # Create flow comparison leaderboard
#     if flow_evaluations:
#         print("🏆 Creating flow comparison leaderboard...")
#         leaderboard_ref = await create_leaderboard(flow_evaluations, "Flow Type Comparison")
#         print(f"✅ Created flow comparison leaderboard")
#         print(f"📊 Check your Weave dashboard for the flow comparison")
#         return leaderboard_ref
#
#     return None


async def main():
    """Main demo function."""
    print("🤖 Weave Leaderboard Integration Demo")
    print("="*60)
    
    print("This demo shows how to integrate Weave leaderboards with your evaluation system:")
    print("• Simple Weave evaluations with custom scorers")
    print("• Leaderboard creation for model comparison")
    print("• Flow-specific evaluation comparisons")
    print("• Custom evaluation scenarios")
    
    # Run different demos
    demos = [
        ("Simple Weave Evaluation", demo_evaluation),
        # ("Custom Evaluations", demo_custom_evaluations),
        # ("Flow Comparison", demo_flow_comparison),
        # ("Leaderboard Creation", demo_leaderboard_creation),
    ]
    
    print(f"\n🚀 Running demos...")
    
    results = []
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            print(f"\n{'='*20} {name} ({i}/{len(demos)}) {'='*20}")
            result = await demo_func()
            results.append((name, result))
            
                
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, None))
    
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

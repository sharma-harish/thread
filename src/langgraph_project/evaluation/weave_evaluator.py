"""
Simple Weave Integration for Multi-Agent Evaluation

Based on the Weave leaderboard documentation, this provides a clean integration
for creating leaderboards and evaluations.
"""

import json
import weave
from weave import Evaluation, Model, op
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import random


# Initialize Weave
weave.init('iams-harish-na/thread')


class FlowType(Enum):
    """Different agent flows in the system."""
    USER = "user"
    DOCUMENTATION = "documentation" 
    GENERAL = "general"
    MORE_INFO = "more-info"


@dataclass
class TestCase:
    """A simple test case."""
    question: str
    expected_flow: FlowType
    expected_response: Optional[str] = None
    expected_entities: Optional[List[str]] = None


# Weave scorers - following the documentation pattern
@op
def flow_accuracy_scorer(question: str, expected_flow: str, model_output: dict) -> Dict[str, Any]:
    """Score flow classification accuracy."""
    actual_flow = model_output.get('flow', '')
    return {
        "correct": expected_flow == actual_flow,
        "expected": expected_flow,
        "actual": actual_flow,
        "score": 1.0 if expected_flow == actual_flow else 0.0
    }


@op
def response_quality_scorer(question: str, expected_response: str, model_output: dict) -> Dict[str, Any]:
    """Score response quality based on relevance."""
    actual_response = model_output.get('response', '')
    
    if not actual_response:
        return {"score": 0.0, "reason": "No response generated"}
    
    # Simple scoring based on expected content presence
    if expected_response and expected_response.lower() in actual_response.lower():
        return {"score": 1.0, "reason": "Expected content found"}
    
    # Check if response is relevant to question
    question_words = set(question.lower().split())
    response_words = set(actual_response.lower().split())
    overlap = len(question_words.intersection(response_words))
    
    if overlap > 0:
        relevance_score = min(1.0, overlap / len(question_words))
        return {"score": relevance_score, "reason": f"Response relevance: {relevance_score:.2f}"}
    
    return {"score": 0.3, "reason": "Low relevance to question"}


@op
def entity_extraction_scorer(question: str, expected_entities: List[str], model_output: dict) -> Dict[str, Any]:
    """Score entity extraction accuracy."""
    actual_response = model_output.get('response', '')
    
    if not expected_entities:
        return {"score": 1.0, "reason": "No entities expected"}
    
    found_entities = []
    for entity in expected_entities:
        if entity.lower() in actual_response.lower():
            found_entities.append(entity)
    
    accuracy = len(found_entities) / len(expected_entities)
    
    return {
        "score": accuracy,
        "expected_entities": expected_entities,
        "found_entities": found_entities,
        "reason": f"Found {len(found_entities)}/{len(expected_entities)} entities"
    }


# Create a simple model function instead of a class
@op
async def multi_agent_predict(question: str, expected_flow: str = None) -> Dict[str, Any]:
    """Predict using the multi-agent system."""
    from langchain_core.messages import HumanMessage
    from ..state import Router
    from ..main import get_or_create_graph
    
    try:
        graph = get_or_create_graph()
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "message": Router(type=expected_flow or "user", logic="")
        }
        
        result = await graph.ainvoke(initial_state)
        
        actual_flow = None
        response = None
        
        if 'message' in result:
            actual_flow = result['message'].get('type')
        
        if 'messages' in result and result['messages']:
            last_message = result['messages'][-1]
            if hasattr(last_message, 'content'):
                response = last_message.content
        
        return {
            'response': response,
            'flow': actual_flow,
            'question': question
        }
    except Exception as e:
        return {
            'response': None,
            'flow': None,
            'question': question,
            'error': str(e)
        }


def load_test_cases(file_path: str) -> List[TestCase]:
    """Load test cases from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    test_cases = []
    
    # Load from different flow categories
    flow_mapping = {
        "user_flow_tests": FlowType.USER,
        "documentation_flow_tests": FlowType.DOCUMENTATION,
        "general_flow_tests": FlowType.GENERAL,
        "more_info_flow_tests": FlowType.MORE_INFO,
        "edge_cases": FlowType.USER  # Default for edge cases
    }
    
    for category, flow_type in flow_mapping.items():
        if category in data:
            for case_data in data[category]:
                test_case = TestCase(
                    question=case_data['question'],
                    expected_flow=flow_type,
                    expected_response=case_data.get('expected_response'),
                    expected_entities=case_data.get('expected_entities', [])
                )
                test_cases.append(test_case)
    
    return test_cases


def create_evaluation_dataset(test_cases: List[TestCase]) -> List[Dict[str, Any]]:
    """Convert test cases to Weave evaluation dataset format."""
    dataset = []
    
    for test_case in test_cases:
        dataset.append({
            'question': test_case.question,
            'expected_flow': test_case.expected_flow.value,
            'expected_response': test_case.expected_response or "",
            'expected_entities': test_case.expected_entities or []
        })
    
    return dataset


async def create_simple_evaluation(test_cases: List[TestCase], name: str = "Multi-Agent Evaluation") -> Evaluation:
    """Create a simple Weave evaluation."""
    
    # Create dataset
    dataset = create_evaluation_dataset(test_cases)
    
    # Create evaluation with scorers
    evaluation = Evaluation(
        name=name,
        dataset=dataset,
        scorers=[
            flow_accuracy_scorer,
            response_quality_scorer,
            entity_extraction_scorer
        ]
    )
    
    return evaluation


async def run_weave_evaluation(test_cases: List[TestCase], model_name: str = "MultiAgentModel") -> Evaluation:
    """Run Weave evaluation and return results."""
    
    # Create evaluation
    evaluation = await create_simple_evaluation(test_cases, f"{model_name} Evaluation")
    
    # Run evaluation with the model function
    await evaluation.evaluate(multi_agent_predict)
    
    return evaluation

async def run_simple_weave_evaluation(batch):
    """Run a simple Weave evaluation with sample data."""
    # Load test cases
    test_cases = load_test_cases("src/langgraph_project/evaluation/datasets/comprehensive_test_cases.json")
    
    # Take a small sample for demo
    sample_cases = test_cases[:batch]
    
    # Run evaluation
    evaluation = await run_weave_evaluation(sample_cases, "WeaveMultiAgent")
    
    print(f"✅ Weave evaluation completed!")
    print(f"📊 Check your Weave dashboard for results")
    
    return evaluation

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_simple_weave_evaluation())

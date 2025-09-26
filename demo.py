#!/usr/bin/env python3
"""
GDP Evaluation Framework Demo
Showcases the evaluation system using free APIs
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import gradio as gr
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.task_loader import KhmerTaskLoader, EvaluationTask
from src.core.evaluator import KhmerModelEvaluator, BaselineEvaluator
from src.core.grader import AutomatedGrader, BilingualGrader
from src.providers.huggingface_provider import HuggingFaceProvider
from src.providers.ollama_provider import OllamaProvider
from src.utils.khmer_utils import KhmerTextProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GDPvalDemo:
    """Main demo application class"""

    def __init__(self):
        """Initialize demo application"""
        self.providers = {}
        self.current_provider = None
        self.evaluator = None
        self.grader = None
        self.khmer_processor = KhmerTextProcessor()
        self.results = []

        # Initialize components
        self._setup_providers()
        self._setup_evaluators()

    def _setup_providers(self):
        """Set up available LLM providers"""
        # HuggingFace provider (free)
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        if hf_api_key or True:  # HF works without API key for public models
            try:
                self.providers["huggingface"] = HuggingFaceProvider(
                    api_key=hf_api_key,
                    model="mistral-7b",
                    rate_limit=30
                )
                logger.info("HuggingFace provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace provider: {e}")

        # Ollama provider (local)
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        try:
            self.providers["ollama"] = OllamaProvider(
                model="llama2",
                base_url=ollama_url
            )
            logger.info("Ollama provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama provider: {e}")

        # Set default provider
        if self.providers:
            self.current_provider = list(self.providers.keys())[0]
            logger.info(f"Default provider set to: {self.current_provider}")

    def _setup_evaluators(self):
        """Set up evaluators and graders"""
        self.evaluator = KhmerModelEvaluator(
            model_name="demo_model",
            khmer_support=True,
            bilingual_mode=True
        )
        self.grader = BilingualGrader()

    async def test_provider_connection(self, provider_name: str) -> Dict[str, Any]:
        """
        Test connection to a provider

        Args:
            provider_name: Name of the provider

        Returns:
            Connection test results
        """
        if provider_name not in self.providers:
            return {
                "status": "error",
                "message": f"Provider {provider_name} not available"
            }

        provider = self.providers[provider_name]

        try:
            # Test connection
            is_connected = await provider.validate_connection()

            if is_connected:
                # Test generation
                response = await provider.generate(
                    "Hello, can you respond in both English and Khmer? សួស្តី",
                    max_tokens=50
                )

                if response.error:
                    return {
                        "status": "partial",
                        "message": f"Connected but generation failed: {response.error}",
                        "provider_info": provider.get_model_info() if hasattr(provider, 'get_model_info') else {}
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Provider connected and working",
                        "test_response": response.text,
                        "latency_ms": response.latency_ms,
                        "provider_info": provider.get_model_info() if hasattr(provider, 'get_model_info') else {}
                    }
            else:
                return {
                    "status": "error",
                    "message": "Failed to connect to provider"
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection test failed: {str(e)}"
            }

    async def evaluate_sample_task(
        self,
        provider_name: str,
        task_text: str,
        task_category: str = "general"
    ) -> Dict[str, Any]:
        """
        Evaluate a sample task

        Args:
            provider_name: Provider to use
            task_text: Task description
            task_category: Task category

        Returns:
            Evaluation results
        """
        if provider_name not in self.providers:
            return {"error": f"Provider {provider_name} not available"}

        provider = self.providers[provider_name]

        # Create a sample task
        sample_task = EvaluationTask(
            task_id=f"demo_{datetime.now().timestamp()}",
            occupation="Demo User",
            category=task_category,
            industry="Demo",
            gdp_contribution_percent=0.0,
            prompt={
                "instruction": task_text,
                "instruction_english": task_text,
                "output_format": "structured"
            },
            reference_files=[],
            grading_criteria=[
                {
                    "criterion_id": "completeness",
                    "criterion_name": "Completeness",
                    "weight": 1.0,
                    "max_score": 10.0
                },
                {
                    "criterion_id": "accuracy",
                    "criterion_name": "Accuracy",
                    "weight": 1.0,
                    "max_score": 10.0
                }
            ],
            metadata={
                "language": "bilingual",
                "difficulty_level": 3,
                "estimated_time_minutes": 10
            }
        )

        try:
            # Generate response
            context = self.evaluator._prepare_context(sample_task)
            response = await provider.generate(context, max_tokens=500, temperature=0.7)

            if response.error:
                return {"error": f"Generation failed: {response.error}"}

            # Create evaluation result
            from src.core.evaluator import EvaluationResult
            eval_result = EvaluationResult(
                task_id=sample_task.task_id,
                model_name=provider_name,
                response=response.text,
                latency_ms=response.latency_ms,
                api_cost=response.cost,
                timestamp=datetime.now().isoformat(),
                metadata=response.metadata
            )

            # Grade the response
            grading_result = self.grader.grade_response(sample_task, eval_result)

            # Analyze Khmer content
            khmer_analysis = self.evaluator.validate_khmer_response(response.text)

            return {
                "task_id": sample_task.task_id,
                "provider": provider_name,
                "model": response.model,
                "response": response.text,
                "grading": {
                    "total_score": grading_result.total_score,
                    "max_score": grading_result.max_score,
                    "percentage": grading_result.percentage,
                    "passed": grading_result.passed(),
                    "feedback": grading_result.feedback
                },
                "khmer_analysis": khmer_analysis,
                "performance": {
                    "latency_ms": response.latency_ms,
                    "tokens_used": response.tokens_used,
                    "cost_usd": response.cost
                }
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": f"Evaluation failed: {str(e)}"}

    async def compare_providers(self, task_text: str) -> pd.DataFrame:
        """
        Compare all available providers on the same task

        Args:
            task_text: Task to evaluate

        Returns:
            Comparison dataframe
        """
        results = []

        for provider_name in self.providers.keys():
            logger.info(f"Evaluating with {provider_name}...")
            result = await self.evaluate_sample_task(provider_name, task_text)

            if "error" not in result:
                results.append({
                    "Provider": provider_name,
                    "Model": result.get("model", "unknown"),
                    "Score (%)": result["grading"]["percentage"],
                    "Passed": "✓" if result["grading"]["passed"] else "✗",
                    "Khmer Content (%)": result["khmer_analysis"]["khmer_percentage"],
                    "Latency (ms)": result["performance"]["latency_ms"],
                    "Tokens": result["performance"]["tokens_used"],
                    "Cost ($)": result["performance"]["cost_usd"]
                })
            else:
                results.append({
                    "Provider": provider_name,
                    "Model": "N/A",
                    "Score (%)": 0,
                    "Passed": "✗",
                    "Khmer Content (%)": 0,
                    "Latency (ms)": 0,
                    "Tokens": 0,
                    "Cost ($)": 0
                })

        return pd.DataFrame(results)

    def create_gradio_interface(self):
        """Create Gradio web interface"""

        async def test_connection(provider):
            """Test provider connection"""
            result = await self.test_provider_connection(provider)
            return json.dumps(result, indent=2, ensure_ascii=False)

        async def evaluate_task(provider, task_text, category):
            """Evaluate a single task"""
            result = await self.evaluate_sample_task(provider, task_text, category)
            return json.dumps(result, indent=2, ensure_ascii=False)

        async def compare_all(task_text):
            """Compare all providers"""
            df = await self.compare_providers(task_text)
            return df

        # Create interface
        with gr.Blocks(title="GDPval Khmer Evaluation Demo") as demo:
            gr.Markdown(
                """
                # 🇰🇭 GDPval Khmer Language Model Evaluation Demo

                This demo showcases the GDP evaluation framework using **free** LLM APIs.
                Test different providers and evaluate their performance on Khmer-English bilingual tasks.
                """
            )

            with gr.Tab("Test Connection"):
                gr.Markdown("## Test Provider Connections")
                with gr.Row():
                    provider_dropdown = gr.Dropdown(
                        choices=list(self.providers.keys()),
                        value=self.current_provider if self.current_provider else None,
                        label="Select Provider"
                    )
                    test_btn = gr.Button("Test Connection", variant="primary")

                connection_output = gr.Code(label="Connection Test Results", language="json")

                test_btn.click(
                    fn=test_connection,
                    inputs=[provider_dropdown],
                    outputs=[connection_output]
                )

            with gr.Tab("Evaluate Task"):
                gr.Markdown("## Evaluate a Single Task")
                with gr.Row():
                    with gr.Column():
                        eval_provider = gr.Dropdown(
                            choices=list(self.providers.keys()),
                            value=self.current_provider if self.current_provider else None,
                            label="Provider"
                        )
                        eval_category = gr.Dropdown(
                            choices=["finance", "technology", "healthcare", "agriculture", "general"],
                            value="general",
                            label="Task Category"
                        )
                        eval_task = gr.Textbox(
                            label="Task Description",
                            placeholder="Enter a task in English or Khmer...",
                            value="Analyze the economic impact of mobile banking adoption in rural Cambodia. វិភាគផលប៉ះពាល់សេដ្ឋកិច្ចនៃការប្រើប្រាស់ធនាគារតាមទូរស័ព្ទនៅតំបន់ជនបទកម្ពុជា។",
                            lines=3
                        )
                        eval_btn = gr.Button("Evaluate", variant="primary")

                eval_output = gr.Code(label="Evaluation Results", language="json")

                eval_btn.click(
                    fn=evaluate_task,
                    inputs=[eval_provider, eval_task, eval_category],
                    outputs=[eval_output]
                )

            with gr.Tab("Compare Providers"):
                gr.Markdown("## Compare All Providers")
                compare_task = gr.Textbox(
                    label="Task for Comparison",
                    placeholder="Enter a task to compare across all providers...",
                    value="Explain the importance of GDP growth for Cambodia's development. ពន្យល់ពីសារៈសំខាន់នៃកំណើន GDP សម្រាប់ការអភិវឌ្ឍន៍កម្ពុជា។",
                    lines=3
                )
                compare_btn = gr.Button("Compare All Providers", variant="primary")

                comparison_output = gr.Dataframe(label="Provider Comparison Results")

                compare_btn.click(
                    fn=compare_all,
                    inputs=[compare_task],
                    outputs=[comparison_output]
                )

            with gr.Tab("Sample Tasks"):
                gr.Markdown(
                    """
                    ## Sample Evaluation Tasks

                    ### Finance / ហិរញ្ញវត្ថុ
                    - Analyze Q3 financial performance of microfinance institutions in Cambodia
                    - Calculate ROI for agricultural loans in Battambang province
                    - Assess credit risk for SME lending in tourism sector

                    ### Technology / បច្ចេកវិទ្យា
                    - Design a mobile app for Khmer language learning
                    - Implement a QR payment system for local markets
                    - Optimize database queries for e-commerce platform

                    ### Healthcare / សុខភាព
                    - Create a vaccination tracking system for rural clinics
                    - Analyze malaria prevention program effectiveness
                    - Design telemedicine solution for remote provinces

                    ### Agriculture / កសិកម្ម
                    - Optimize rice yield predictions using weather data
                    - Design irrigation monitoring system using IoT
                    - Analyze cashew nut export market trends
                    """
                )

            with gr.Tab("Settings"):
                gr.Markdown(
                    """
                    ## Configuration

                    ### Available Providers:
                    - **HuggingFace**: Free tier, 30 requests/minute
                    - **Ollama**: Local execution, unlimited requests
                    - **Google Gemini**: Free tier, 60 requests/minute (if configured)

                    ### API Keys:
                    Set in `.env` file or environment variables:
                    - `HUGGINGFACE_API_KEY` (optional for public models)
                    - `GEMINI_API_KEY` (required for Gemini)
                    - `OLLAMA_URL` (default: http://localhost:11434)
                    """
                )

        return demo


def main():
    """Main function to run the demo"""
    # Initialize demo
    demo_app = GDPvalDemo()

    # Check if any providers are available
    if not demo_app.providers:
        logger.error("No providers available. Please check your configuration.")
        logger.info("Make sure to:")
        logger.info("1. Set API keys in .env file")
        logger.info("2. Install and run Ollama for local execution")
        sys.exit(1)

    # Create and launch Gradio interface
    interface = demo_app.create_gradio_interface()

    # Get server settings from environment
    server_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

    logger.info(f"Starting demo server on {server_name}:{server_port}")
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,  # Set to True to create a public link
        debug=True
    )


if __name__ == "__main__":
    # Run the demo
    main()
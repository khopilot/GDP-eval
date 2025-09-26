"""
Report Generator - Enterprise Grade
Generates comprehensive evaluation reports in multiple formats
OpenAI/Anthropic-level evaluation standards
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict
import logging

from .evaluation_orchestrator import ComprehensiveEvaluationResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Enterprise-grade report generation system
    Creates detailed reports in multiple formats
    """

    def __init__(self, output_dir: str = "results/reports"):
        self.output_dir = output_dir
        self.ensure_output_directory()

    def ensure_output_directory(self):
        """Ensure output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/json", exist_ok=True)
        os.makedirs(f"{self.output_dir}/html", exist_ok=True)
        os.makedirs(f"{self.output_dir}/text", exist_ok=True)

    def generate_json_report(self, result: ComprehensiveEvaluationResult, filename: Optional[str] = None) -> str:
        """
        Generate JSON format report

        Args:
            result: Comprehensive evaluation results
            filename: Optional custom filename

        Returns:
            Path to generated JSON report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = result.metadata.get('model_name', 'unknown').replace(' ', '_')
            filename = f"evaluation_report_{model_name}_{timestamp}.json"

        filepath = os.path.join(self.output_dir, "json", filename)

        try:
            # Convert result to serializable format
            serializable_result = self._make_serializable(result)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"JSON report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            raise

    def generate_html_report(self, result: ComprehensiveEvaluationResult, filename: Optional[str] = None) -> str:
        """
        Generate HTML format report

        Args:
            result: Comprehensive evaluation results
            filename: Optional custom filename

        Returns:
            Path to generated HTML report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = result.metadata.get('model_name', 'unknown').replace(' ', '_')
            filename = f"evaluation_report_{model_name}_{timestamp}.html"

        filepath = os.path.join(self.output_dir, "html", filename)

        try:
            html_content = self._generate_html_content(result)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise

    def generate_text_report(self, result: ComprehensiveEvaluationResult, filename: Optional[str] = None) -> str:
        """
        Generate text format report

        Args:
            result: Comprehensive evaluation results
            filename: Optional custom filename

        Returns:
            Path to generated text report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = result.metadata.get('model_name', 'unknown').replace(' ', '_')
            filename = f"evaluation_report_{model_name}_{timestamp}.txt"

        filepath = os.path.join(self.output_dir, "text", filename)

        try:
            # Import orchestrator to use its report generation
            from .evaluation_orchestrator import EvaluationOrchestrator
            orchestrator = EvaluationOrchestrator()
            text_content = orchestrator.generate_evaluation_report(result)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)

            logger.info(f"Text report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
            raise

    def generate_all_formats(self, result: ComprehensiveEvaluationResult, base_filename: Optional[str] = None) -> Dict[str, str]:
        """
        Generate reports in all supported formats

        Args:
            result: Comprehensive evaluation results
            base_filename: Base filename (without extension)

        Returns:
            Dictionary mapping format to filepath
        """
        if not base_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = result.metadata.get('model_name', 'unknown').replace(' ', '_')
            base_filename = f"evaluation_report_{model_name}_{timestamp}"

        generated_files = {}

        try:
            # Generate JSON report
            json_file = self.generate_json_report(result, f"{base_filename}.json")
            generated_files["json"] = json_file

            # Generate HTML report
            html_file = self.generate_html_report(result, f"{base_filename}.html")
            generated_files["html"] = html_file

            # Generate text report
            text_file = self.generate_text_report(result, f"{base_filename}.txt")
            generated_files["text"] = text_file

            logger.info(f"All format reports generated for {base_filename}")
            return generated_files

        except Exception as e:
            logger.error(f"Failed to generate all format reports: {e}")
            raise

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        elif hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def _generate_html_content(self, result: ComprehensiveEvaluationResult) -> str:
        """Generate HTML content for the report"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Model Evaluation Report - {result.metadata.get('model_name', 'Unknown')}</title>
    <style>
        {self._get_html_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI Model Evaluation Report</h1>
            <div class="subtitle">Enterprise-Grade Assessment • GDP-eval Framework</div>
        </header>

        <section class="executive-summary">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card overall-score">
                    <h3>Overall Score</h3>
                    <div class="score-large">{result.overall_score:.1f}</div>
                    <div class="score-label">out of 100</div>
                </div>
                <div class="summary-card confidence">
                    <h3>Confidence Level</h3>
                    <div class="score-medium">{result.confidence:.2f}</div>
                    <div class="score-label">reliability index</div>
                </div>
                <div class="summary-card risk">
                    <h3>Risk Assessment</h3>
                    <div class="risk-badge risk-{result.risk_assessment.lower()}">{result.risk_assessment}</div>
                </div>
            </div>
        </section>

        <section class="model-info">
            <h2>🔧 Model Information</h2>
            <div class="info-grid">
                <div class="info-item">
                    <strong>Model Name:</strong> {result.metadata.get('model_name', 'Unknown')}
                </div>
                <div class="info-item">
                    <strong>Evaluation Suite:</strong> {result.metadata.get('evaluation_suite', 'Unknown').upper()}
                </div>
                <div class="info-item">
                    <strong>Evaluation Date:</strong> {result.metadata.get('start_time', 'Unknown')}
                </div>
                <div class="info-item">
                    <strong>Duration:</strong> {result.metadata.get('duration_seconds', 0):.1f} seconds
                </div>
                <div class="info-item">
                    <strong>Total Tests:</strong> {result.metadata.get('total_tests', 0)}
                </div>
                <div class="info-item">
                    <strong>Modules:</strong> {', '.join(result.metadata.get('modules_evaluated', []))}
                </div>
            </div>
        </section>

        <section class="evaluation-breakdown">
            <h2>📈 Evaluation Breakdown</h2>
            <div class="breakdown-grid">
                {self._generate_breakdown_html(result.evaluation_summary)}
            </div>
        </section>

        <section class="recommendations">
            <h2>💡 Recommendations</h2>
            {self._generate_recommendations_html(result.recommendations)}
        </section>

        <section class="detailed-results">
            <h2>📋 Detailed Results</h2>
            {self._generate_detailed_results_html(result.detailed_results)}
        </section>

        <footer>
            <p>Generated by GDP-eval Framework • Enterprise-Grade AI Evaluation</p>
            <p>Contact: Nicolas Delrieu, AI Consultant • +855 92 332 554</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        section {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }

        h2 {
            color: #2d3748;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .summary-card {
            text-align: center;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        .summary-card h3 {
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #4a5568;
        }

        .overall-score {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
        }

        .confidence {
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
        }

        .risk {
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
            color: white;
        }

        .score-large {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .score-medium {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .score-label {
            font-size: 0.9em;
            opacity: 0.8;
        }

        .risk-badge {
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.2em;
        }

        .risk-low { background: #48bb78; color: white; }
        .risk-medium { background: #ed8936; color: white; }
        .risk-high { background: #e53e3e; color: white; }
        .risk-critical { background: #9b2c2c; color: white; }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }

        .info-item {
            padding: 15px;
            background: #f7fafc;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }

        .breakdown-grid {
            display: grid;
            gap: 15px;
        }

        .breakdown-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .breakdown-name {
            font-weight: bold;
            font-size: 1.1em;
            text-transform: capitalize;
        }

        .breakdown-score {
            font-size: 1.3em;
            font-weight: bold;
            color: #2d3748;
        }

        .breakdown-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }

        .status-pass { background: #c6f6d5; color: #22543d; }
        .status-attention { background: #fef5e7; color: #c05621; }
        .status-fail { background: #fed7d7; color: #c53030; }

        .recommendations ul {
            list-style: none;
        }

        .recommendations li {
            padding: 15px;
            margin-bottom: 10px;
            background: #fef5e7;
            border-left: 4px solid #ed8936;
            border-radius: 5px;
        }

        .detailed-section {
            margin-bottom: 25px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .detailed-section h4 {
            color: #2d3748;
            margin-bottom: 10px;
            text-transform: uppercase;
            font-size: 1.1em;
        }

        footer {
            text-align: center;
            padding: 30px 20px;
            background: #2d3748;
            color: white;
            border-radius: 10px;
            margin-top: 40px;
        }

        footer p {
            margin-bottom: 5px;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .summary-grid {
                grid-template-columns: 1fr;
            }

            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _generate_breakdown_html(self, evaluation_summary: Dict[str, float]) -> str:
        """Generate HTML for evaluation breakdown"""
        html_parts = []

        for module, score in evaluation_summary.items():
            if score >= 75:
                status = "pass"
                status_text = "✓ PASS"
            elif score >= 60:
                status = "attention"
                status_text = "⚠ ATTENTION"
            else:
                status = "fail"
                status_text = "❌ FAIL"

            html_parts.append(f"""
            <div class="breakdown-item">
                <div>
                    <div class="breakdown-name">{module}</div>
                </div>
                <div>
                    <span class="breakdown-score">{score:.1f}/100</span>
                    <span class="breakdown-status status-{status}">{status_text}</span>
                </div>
            </div>
            """)

        return "".join(html_parts)

    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations"""
        if not recommendations:
            return "<p>No specific recommendations at this time.</p>"

        html_parts = ["<ul>"]
        for rec in recommendations:
            html_parts.append(f"<li>{rec}</li>")
        html_parts.append("</ul>")

        return "".join(html_parts)

    def _generate_detailed_results_html(self, detailed_results: Dict[str, Any]) -> str:
        """Generate HTML for detailed results"""
        html_parts = []

        for module_name, module_result in detailed_results.items():
            if isinstance(module_result, dict) and "error" in module_result:
                html_parts.append(f"""
                <div class="detailed-section">
                    <h4>❌ {module_name.upper()} Module Failed</h4>
                    <p>Error: {module_result['error']}</p>
                </div>
                """)
            else:
                html_parts.append(f"""
                <div class="detailed-section">
                    <h4>{module_name.upper()} Module Results</h4>
                    <p>Detailed analysis completed with comprehensive testing.</p>
                """)

                # Add specific details if available
                if hasattr(module_result, 'score'):
                    html_parts.append(f"<p><strong>Score:</strong> {module_result.score:.1f}/100</p>")
                if hasattr(module_result, 'confidence'):
                    html_parts.append(f"<p><strong>Confidence:</strong> {module_result.confidence:.2f}</p>")

                html_parts.append("</div>")

        return "".join(html_parts)

    def generate_comparison_report(self, results: List[ComprehensiveEvaluationResult], filename: Optional[str] = None) -> str:
        """
        Generate comparison report for multiple models

        Args:
            results: List of evaluation results to compare
            filename: Optional custom filename

        Returns:
            Path to generated comparison report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.html"

        filepath = os.path.join(self.output_dir, "html", filename)

        try:
            html_content = self._generate_comparison_html(results)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Comparison report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            raise

    def _generate_comparison_html(self, results: List[ComprehensiveEvaluationResult]) -> str:
        """Generate HTML for model comparison report"""
        if not results:
            return "<html><body><h1>No results to compare</h1></body></html>"

        # Create comparison table
        comparison_table = self._create_comparison_table(results)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report - GDP-eval</title>
    <style>{self._get_html_styles()}</style>
    <style>
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .comparison-table th, .comparison-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        .comparison-table th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        .comparison-table tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .best-score {{
            background: #c6f6d5 !important;
            font-weight: bold;
        }}
        .worst-score {{
            background: #fed7d7 !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 Model Comparison Report</h1>
            <div class="subtitle">Enterprise-Grade AI Model Evaluation Comparison</div>
        </header>

        <section>
            <h2>📊 Performance Comparison</h2>
            {comparison_table}
        </section>

        <footer>
            <p>Generated by GDP-eval Framework • Enterprise-Grade AI Evaluation</p>
            <p>Contact: Nicolas Delrieu, AI Consultant • +855 92 332 554</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _create_comparison_table(self, results: List[ComprehensiveEvaluationResult]) -> str:
        """Create comparison table HTML"""
        if not results:
            return "<p>No results to compare</p>"

        # Get all unique modules
        all_modules = set()
        for result in results:
            all_modules.update(result.evaluation_summary.keys())
        all_modules = sorted(all_modules)

        # Create table header
        table_html = ['<table class="comparison-table">']
        table_html.append('<thead><tr>')
        table_html.append('<th>Model</th>')
        table_html.append('<th>Overall Score</th>')
        table_html.append('<th>Risk Assessment</th>')
        for module in all_modules:
            table_html.append(f'<th>{module.capitalize()}</th>')
        table_html.append('</tr></thead>')

        # Find best scores for highlighting
        best_overall = max(results, key=lambda r: r.overall_score)
        best_scores = {}
        for module in all_modules:
            module_scores = [r.evaluation_summary.get(module, 0) for r in results]
            if module_scores:
                best_scores[module] = max(module_scores)

        # Create table rows
        table_html.append('<tbody>')
        for result in results:
            table_html.append('<tr>')

            # Model name
            model_name = result.metadata.get('model_name', 'Unknown')
            table_html.append(f'<td><strong>{model_name}</strong></td>')

            # Overall score
            overall_class = 'best-score' if result.overall_score == best_overall.overall_score else ''
            table_html.append(f'<td class="{overall_class}">{result.overall_score:.1f}</td>')

            # Risk assessment
            risk_class = f'risk-{result.risk_assessment.lower()}'
            table_html.append(f'<td><span class="risk-badge {risk_class}">{result.risk_assessment}</span></td>')

            # Module scores
            for module in all_modules:
                score = result.evaluation_summary.get(module, 0)
                score_class = 'best-score' if score == best_scores.get(module, 0) else ''
                table_html.append(f'<td class="{score_class}">{score:.1f}</td>')

            table_html.append('</tr>')

        table_html.append('</tbody></table>')

        return ''.join(table_html)
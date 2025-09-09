"""
Evaluation framework for CSP causal structure assessment.
Provides comprehensive metrics computation and representation analysis.
"""

from .hooks import (
    ModelEvaluator,
    RepresentationExtractor,
    CSPMetricsComputer
)

__all__ = [
    'ModelEvaluator',
    'RepresentationExtractor',
    'CSPMetricsComputer'
]

# Evaluation utilities
def evaluate_model(model, dataloader, scenario, output_dir=None):
    """
    Convenience function for model evaluation.
    
    Args:
        model: Trained CARL model
        dataloader: Data loader for evaluation
        scenario: Model scenario
        output_dir: Directory for saving results
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(
        model=model,
        scenario=scenario,
        output_dir=output_dir
    )
    
    return evaluator.evaluate(dataloader)

def extract_representations(model, dataloader, representation_types=None):
    """
    Convenience function for representation extraction.
    
    Args:
        model: Trained CARL model
        dataloader: Data loader
        representation_types: Types of representations to extract
        
    Returns:
        Dictionary of extracted representations
    """
    extractor = RepresentationExtractor(model)
    return extractor.extract_representations(
        dataloader,
        representation_types=representation_types
    )
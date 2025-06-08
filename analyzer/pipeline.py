import logging
from typing import Dict, List, Any, Optional

# Import analysis modules
from .statistical import perform_statistical_analysis
from .linguistic import perform_linguistic_analysis
from .semantic import perform_semantic_analysis
from .structural import perform_structural_analysis

# Импорт AI-фактчекера
from .ai_checker import ai_fact_check

import asyncio

logger = logging.getLogger(__name__)

async def analyze_text(text: str, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs a comprehensive multi-level analysis of the news text, including AI-based fact-checking.
    """
    logger.info("Starting comprehensive text analysis")

    results = {
        "statistical": {},
        "linguistic": {},
        "semantic": {},
        "structural": {},
        "credibility_score": 0.5,
        "key_claims": [],
        "suspicious_fragments": [],
        "ai_fact_check": None
    }

    try:
        # Step 1: Statistical analysis
        try:
            logger.info("Performing statistical analysis")
            results["statistical"] = await perform_statistical_analysis(text)
        except Exception as e:
            logger.error(f"Error during statistical analysis: {e}")
            results["statistical"] = {
                "credibility_score": 0.5,
                "error": str(e)
            }

        # Step 2: Linguistic analysis
        try:
            logger.info("Performing linguistic analysis")
            results["linguistic"] = await perform_linguistic_analysis(text)
        except Exception as e:
            logger.error(f"Error during linguistic analysis: {e}")
            results["linguistic"] = {
                "credibility_score": 0.5,
                "error": str(e)
            }

        # Step 3: Semantic analysis
        try:
            logger.info("Performing semantic analysis")
            results["semantic"] = await perform_semantic_analysis(text)
        except Exception as e:
            logger.error(f"Error during semantic analysis: {e}")
            results["semantic"] = {
                "credibility_score": 0.5,
                "error": str(e)
            }

        # Step 4: Structural analysis
        try:
            logger.info("Performing structural analysis")
            results["structural"] = await perform_structural_analysis(text, url)
        except Exception as e:
            logger.error(f"Error during structural analysis: {e}")
            results["structural"] = {
                "credibility_score": 0.5,
                "error": str(e)
            }

        # Step 5: Extract key claims for fact checking
        try:
            key_claims = extract_key_claims(results["semantic"])
            results["key_claims"] = key_claims
        except Exception as e:
            logger.error(f"Error extracting key claims: {e}")
            results["key_claims"] = []

        # Step 6: Collect suspicious fragments
        try:
            suspicious_fragments = identify_suspicious_fragments(
                text,
                results["statistical"],
                results["linguistic"],
                results["semantic"],
                results["structural"]
            )
            results["suspicious_fragments"] = suspicious_fragments
        except Exception as e:
            logger.error(f"Error identifying suspicious fragments: {e}")
            results["suspicious_fragments"] = []

        # Step 7: Calculate overall credibility score
        try:
            credibility_score = calculate_credibility_score(
                results["statistical"],
                results["linguistic"],
                results["semantic"],
                results["structural"]
            )
            results["credibility_score"] = credibility_score
        except Exception as e:
            logger.error(f"Error calculating credibility score: {e}")
            results["credibility_score"] = 0.5

        # Step 8: AI-фактчекер (Qwen3-235B через OpenRouter)
        try:
            logger.info("Performing AI-based fact checking (Qwen3-235B)")
            # Запускать синхронно (чтобы не блокировать event loop), можно через run_in_executor
            loop = asyncio.get_event_loop()
            ai_result = await loop.run_in_executor(None, ai_fact_check, text)
            results["ai_fact_check"] = ai_result
        except Exception as e:
            logger.error(f"Error during AI fact checking: {e}")
            results["ai_fact_check"] = "Ошибка AI-фактчекера: " + str(e)

        return results

    except Exception as e:
        logger.error(f"Unexpected error during text analysis: {e}")
        results["error"] = str(e)
        return results

def extract_key_claims(semantic_results: Dict[str, Any]) -> List[str]:
    """Extract key claims from semantic analysis for fact checking"""
    claims = semantic_results.get("identified_claims", [])
    return claims if isinstance(claims, list) else []

def calculate_credibility_score(
    statistical_results: Dict[str, Any],
    linguistic_results: Dict[str, Any],
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any]
) -> float:
    """
    Calculate an overall credibility score based on all analysis results.
    Returns a score between 0 (likely fake) and 1 (likely credible).
    """
    statistical_score = statistical_results.get("credibility_score", 0.5)
    linguistic_score = linguistic_results.get("credibility_score", 0.5)
    semantic_score = semantic_results.get("credibility_score", 0.5)
    structural_score = structural_results.get("credibility_score", 0.5)

    weights = [0.25, 0.3, 0.3, 0.15]  # Adjust based on importance
    overall_score = sum([
        statistical_score * weights[0],
        linguistic_score * weights[1],
        semantic_score * weights[2],
        structural_score * weights[3]
    ])
    return overall_score

def identify_suspicious_fragments(
    text: str,
    statistical_results: Dict[str, Any],
    linguistic_results: Dict[str, Any],
    semantic_results: Dict[str, Any],
    structural_results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Identify suspicious text fragments based on analysis results.
    """
    suspicious_fragments = []

    linguistic_fragments = linguistic_results.get("suspicious_fragments", [])
    if isinstance(linguistic_fragments, list):
        suspicious_fragments.extend(linguistic_fragments)

    semantic_fragments = semantic_results.get("suspicious_fragments", [])
    if isinstance(semantic_fragments, list):
        suspicious_fragments.extend(semantic_fragments)

    return suspicious_fragments

"""
Multi-turn inference prompt templates and formatting utilities
"""
from typing import Dict, Optional


class PromptTemplates:
    """Prompt template management class"""
    
    # ═════════════════════════════════════════════
    # Scene Description
    # ═════════════════════════════════════════════
    SCENE_DESCRIPTION = """Suppose you are driving, generate a description of the driving scene which includes the key factors for driving planning, including the traffic conditions, weather, time of day and road conditions, traffic signs, and traffic lights that affect the driving of the ego vehicle if it exists, indicating smooth surfaces or the presence of obstacles; The description should be concise, and accurate to facilitate informed decision-making. Please make sure the traffic light colors you provide are accurate; otherwise, give ‘unknown.’"""
    
    # ═════════════════════════════════════════════
    # Scene Analysis
    # ═════════════════════════════════════════════
    SCENE_ANALYSIS = """Based on the scene description you just provided, please identify the critical objects that are most important for the driving decision. For each object, please describe its action or state (e.g., moving, parked, braking) and explain its potential influence on our (ego vehicle's) driving plan."""
    # SCENE_ANALYSIS = """You just provided a scene description. Before proceeding, critically verify your last response against the image. Are there any inconsistencies or errors in your description when compared to the visual evidence (especially traffic lights, object states)? Based on this visual verification, provide an updated and corrected Scene Analysis. Identify the critical objects, their states, and their potential influence."""

    # ═════════════════════════════════════════════
    # Planning
    # ═════════════════════════════════════════════
    PLANNING = """Now, based on all your analysis so far, what should our driving behavior be? Please provide the expert meta-decision (e.g., accelerate, decelerate, change lane left) and explain the reasoning for this plan, considering other participants, road conditions, and traffic lights."""
    # PLANNING = """You have just provided a verified analysis. Now, using both that analysis and a final look at the image, determine the safest driving plan. Your plan must be fully consistent with the visual reality (especially traffic lights and pedestrian movements)."""

    @staticmethod
    def format_scene_description(image_placeholder: str = "<image>") -> str:
        """Scene description prompt formatting"""
        return f"{image_placeholder}\n{PromptTemplates.SCENE_DESCRIPTION}"


class PromptConfig:
    """Generation configuration for each stage"""
    
    SCENE_DESCRIPTION = {
        "max_new_tokens": 400,
        "do_sample": False,
        "num_beams": 1,
        "temperature": 0.7,
    }
    
    SCENE_ANALYSIS = {
        "max_new_tokens": 400,
        "do_sample": False,
        "num_beams": 1,
        "temperature": 0.7,
    }
    
    PLANNING = {
        "max_new_tokens": 400,
        "do_sample": False,
        "num_beams": 1,
        "temperature": 0.7,
    }
"""
Response style management for the AI agent system.

This module provides different communication styles that can be adapted
based on user preferences or automatically detected from conversation context.
"""

from typing import Dict, Optional
from enum import Enum


class ResponseStyle(Enum):
    """Enumeration of available response styles."""
    CASUAL = "casual"
    TECHNICAL = "technical"
    BRIEF = "brief"
    EDUCATIONAL = "educational"


class StyleManager:
    """Manages response styles and style detection/switching."""
    
    def __init__(self):
        self.current_style = ResponseStyle.CASUAL
        self.style_prompts = {
            ResponseStyle.CASUAL: {
                "name": "Casual & Friendly",
                "description": "Conversational, approachable, clean formatting",
                "prompt_modifier": """
                Respond in a casual, friendly, conversational tone. Be approachable and warm
                while still being informative. Use everyday language and avoid overly technical
                jargon unless specifically asked.

                IMPORTANT FORMATTING RULES:
                - Do NOT use emojis in responses
                - Keep responses concise - prioritize clarity over length
                - Use clean bullet points for lists, not decorated headers
                - For data tables (trending tokens, portfolio), show top 5-7 items max
                - Avoid excessive punctuation (!!!, ???) and all-caps
                - One insight or recommendation at the end, not a long list
                """,
                "example_phrases": ["Hey there!", "That's great!", "Let me break this down for you", "Hope this helps!"]
            },
            ResponseStyle.TECHNICAL: {
                "name": "Technical & Detailed",
                "description": "Precise, data-focused analysis with technical depth",
                "prompt_modifier": """
                Respond in a technical, precise manner. Provide specific data points, metrics,
                and technical terminology. Focus on accuracy and depth.

                FORMATTING RULES:
                - Do NOT use emojis
                - Lead with key metrics and data
                - Use clean tables or bullet points
                - Keep analysis focused and scannable
                """,
                "example_phrases": ["Based on the data", "The metrics indicate", "From a protocol perspective"]
            },
            ResponseStyle.BRIEF: {
                "name": "Brief & Actionable",
                "description": "Ultra-concise, to-the-point responses",
                "prompt_modifier": """
                Keep responses extremely concise. Focus only on the key points and main
                takeaways. Be direct and efficient.

                FORMATTING RULES:
                - Do NOT use emojis
                - Maximum 3-5 bullet points
                - One sentence summary if possible
                - No pleasantries or filler
                """,
                "example_phrases": ["Key points:", "Bottom line:", "Summary:"]
            },
            ResponseStyle.EDUCATIONAL: {
                "name": "Educational & Learning-Focused",
                "description": "Clear step-by-step explanations for learning",
                "prompt_modifier": """
                Focus on teaching and education. Break down complex concepts into digestible
                steps. Use analogies and examples to clarify.

                FORMATTING RULES:
                - Do NOT use emojis
                - Use numbered steps for processes
                - Keep explanations concise but clear
                - One concept at a time
                """,
                "example_phrases": ["Let me explain:", "Think of it this way:", "Here's how it works:"]
            }
        }
    
    def get_style_info(self, style: ResponseStyle) -> Dict[str, str]:
        """Get information about a specific style."""
        return self.style_prompts.get(style, {})
    
    def get_current_style(self) -> ResponseStyle:
        """Get the currently active response style."""
        return self.current_style
    
    def set_style(self, style: ResponseStyle) -> bool:
        """Set the response style."""
        if style in self.style_prompts:
            self.current_style = style
            return True
        return False
    
    def detect_style_from_message(self, message: str) -> Optional[ResponseStyle]:
        """
        Attempt to detect preferred style from user message patterns.
        Returns None if no clear style preference is detected.
        """
        message_lower = message.lower()
        
        # Technical style indicators
        technical_keywords = [
            "technical", "detailed", "analysis", "data", "metrics", "protocol", 
            "implementation", "architecture", "algorithm", "specifics", "precise"
        ]
        
        # Brief style indicators  
        brief_keywords = [
            "brief", "quick", "summary", "tldr", "short", "concise", 
            "bottom line", "key points", "main"
        ]
        
        # Educational style indicators
        educational_keywords = [
            "explain", "how does", "what is", "why", "teach me", "learn", 
            "understand", "help me", "step by step", "guide"
        ]
        
        # Casual style indicators (emojis, casual language)
        casual_indicators = [
            "hey", "hi", "awesome", "cool", "thanks", "thx", 
            "😊", "👍", "🚀", "💰", "🎯"
        ]
        
        # Count indicators for each style
        technical_score = sum(1 for keyword in technical_keywords if keyword in message_lower)
        brief_score = sum(1 for keyword in brief_keywords if keyword in message_lower)
        educational_score = sum(1 for keyword in educational_keywords if keyword in message_lower)
        casual_score = sum(1 for indicator in casual_indicators if indicator in message_lower)
        
        # Determine style based on highest score (with minimum threshold)
        scores = {
            ResponseStyle.TECHNICAL: technical_score,
            ResponseStyle.BRIEF: brief_score,
            ResponseStyle.EDUCATIONAL: educational_score,
            ResponseStyle.CASUAL: casual_score
        }
        
        max_style = max(scores, key=scores.get)
        max_score = scores[max_style]
        
        # Only return a style if there's a clear preference (score >= 2)
        if max_score >= 2:
            return max_style
        
        return None
    
    def parse_style_command(self, message: str) -> Optional[ResponseStyle]:
        """
        Parse explicit style commands from user messages.
        Commands: /style casual, /style technical, /style brief, /style educational
        """
        message_lower = message.lower().strip()
        
        if message_lower.startswith('/style '):
            style_name = message_lower.replace('/style ', '').strip()
            
            style_mapping = {
                'casual': ResponseStyle.CASUAL,
                'friendly': ResponseStyle.CASUAL,
                'conversational': ResponseStyle.CASUAL,
                'technical': ResponseStyle.TECHNICAL,
                'detailed': ResponseStyle.TECHNICAL,
                'precise': ResponseStyle.TECHNICAL,
                'brief': ResponseStyle.BRIEF,
                'short': ResponseStyle.BRIEF,
                'concise': ResponseStyle.BRIEF,
                'educational': ResponseStyle.EDUCATIONAL,
                'learning': ResponseStyle.EDUCATIONAL,
                'teaching': ResponseStyle.EDUCATIONAL
            }
            
            return style_mapping.get(style_name)
        
        return None
    
    def get_style_modifier_prompt(self, style: Optional[ResponseStyle] = None) -> str:
        """Get the prompt modifier for the specified or current style."""
        style_to_use = style or self.current_style
        return self.style_prompts.get(style_to_use, {}).get("prompt_modifier", "")
    
    def get_available_styles(self) -> Dict[str, Dict[str, str]]:
        """Get information about all available styles."""
        return {
            style.value: {
                "name": info["name"],
                "description": info["description"]
            }
            for style, info in self.style_prompts.items()
        }
    
    def format_style_help(self) -> str:
        """Generate a help message about available styles."""
        help_text = "**Available Response Styles:**\n\n"

        for style, info in self.style_prompts.items():
            current_marker = " (current)" if style == self.current_style else ""
            help_text += f"**{info['name']}**{current_marker}\n"
            help_text += f"- {info['description']}\n"
            help_text += f"- Command: `/style {style.value}`\n\n"

        help_text += "**Tips:**\n"
        help_text += "- Use `/style [name]` to switch styles\n"
        help_text += "- Style preference is remembered for the conversation\n"

        return help_text

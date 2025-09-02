"""
Persona Management System

This module defines different AI personalities for the agent system,
allowing dynamic switching between communication styles and specializations.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from enum import Enum
import yaml
import os
from pathlib import Path


class PersonaType(str, Enum):
    """Available persona types"""
    FRIENDLY = "friendly"
    TECHNICAL = "technical"
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"


class Persona(BaseModel):
    """AI Persona definition with system prompt and style guidelines"""
    
    name: str = Field(description="Persona identifier")
    display_name: str = Field(description="Human-readable persona name")
    description: str = Field(description="Brief description of persona characteristics")
    system_prompt: str = Field(description="Core system prompt for this persona")
    
    # Style guidelines
    tone: str = Field(description="Overall communication tone")
    formality: str = Field(description="Level of formality (casual, professional, etc.)")
    technical_depth: str = Field(description="How technical responses should be")
    
    # Response formatting preferences
    use_emojis: bool = Field(default=False, description="Whether to use emojis in responses")
    response_length: str = Field(default="medium", description="Preferred response length (short, medium, long)")
    
    # Specialized knowledge areas
    specializations: List[str] = Field(default_factory=list, description="Areas of specialized knowledge")
    
    def get_system_prompt(self) -> str:
        """Get the complete system prompt for this persona"""
        return self.system_prompt
    
    def format_response_style(self, content: str) -> str:
        """Apply persona-specific formatting to response content"""
        # This is a placeholder for future response formatting logic
        # For now, return content as-is
        return content


class PersonaManager:
    """Manages persona definitions and switching logic"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._personas: Dict[str, Persona] = {}
        self._conversation_personas: Dict[str, str] = {}
        
        # Initialize default personas
        self._initialize_default_personas()
    
    def _initialize_default_personas(self) -> None:
        """Initialize personas by loading from YAML configuration files"""
        self._personas = {}
        
        # Get the personas directory path (relative to the project root)
        current_dir = Path(__file__).parent.parent.parent.parent  # Navigate to project root
        personas_dir = current_dir / "personas"
        
        self.logger.info(f"Loading personas from: {personas_dir}")
        
        # Load all YAML files in the personas directory
        if personas_dir.exists():
            for yaml_file in personas_dir.glob("*.yaml"):
                try:
                    persona = self._load_persona_from_yaml(yaml_file)
                    if persona:
                        self._personas[persona.name] = persona
                        self.logger.info(f"Loaded persona: {persona.name} ({persona.display_name})")
                except Exception as e:
                    self.logger.error(f"Failed to load persona from {yaml_file}: {str(e)}")
        else:
            self.logger.warning(f"Personas directory not found: {personas_dir}")
            # Fallback to a basic friendly persona if no YAML files are found
            self._load_fallback_personas()
        
        # Ensure we have at least one persona
        if not self._personas:
            self.logger.warning("No personas loaded, creating fallback friendly persona")
            self._load_fallback_personas()
    
    def _load_persona_from_yaml(self, yaml_file: Path) -> Optional[Persona]:
        """Load a single persona from a YAML file"""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Enhanced auto-detection keywords from YAML
            auto_detection_keywords = data.get('auto_detection_keywords', [])
            
            # Create the persona object
            persona = Persona(
                name=data['name'],
                display_name=data['display_name'],
                description=data['description'],
                system_prompt=data['system_prompt'].strip(),
                tone=data['tone'],
                formality=data['formality'],
                technical_depth=data['technical_depth'],
                use_emojis=data.get('use_emojis', False),
                response_length=data.get('response_length', 'medium'),
                specializations=data.get('specializations', [])
            )
            
            # Store auto-detection keywords for enhanced persona detection
            if hasattr(self, '_persona_keywords'):
                self._persona_keywords[data['name']] = auto_detection_keywords
            else:
                self._persona_keywords = {data['name']: auto_detection_keywords}
            
            return persona
            
        except Exception as e:
            self.logger.error(f"Error loading persona from {yaml_file}: {str(e)}")
            return None
    
    def _load_fallback_personas(self) -> None:
        """Load minimal fallback personas if YAML files are not available"""
        friendly_persona = Persona(
            name="friendly",
            display_name="Friendly Crypto Guide",
            description="Approachable and encouraging crypto assistant",
            system_prompt="""You are a friendly and knowledgeable crypto portfolio assistant. 
            Your personality is warm, approachable, and encouraging. Use a conversational, casual tone 
            and be encouraging and supportive. Break down complex information into digestible pieces 
            and always maintain a helpful, patient demeanor.""",
            tone="warm and conversational",
            formality="casual",
            technical_depth="medium",
            use_emojis=True,
            response_length="medium",
            specializations=["portfolio_analysis", "crypto_education"]
        )
        
        self._personas = {"friendly": friendly_persona}
    
    def get_persona(self, name: str) -> Persona:
        """Get persona by name, defaulting to friendly if not found"""
        return self._personas.get(name, self._personas["friendly"])
    
    def has_persona(self, name: str) -> bool:
        """Check if a persona exists"""
        return name in self._personas
    
    def list_personas(self) -> Dict[str, str]:
        """Get a list of available personas with their display names"""
        return {
            name: persona.display_name 
            for name, persona in self._personas.items()
        }
    
    def get_persona_description(self, name: str) -> Optional[str]:
        """Get persona description for help/info purposes"""
        persona = self._personas.get(name)
        return persona.description if persona else None
    
    def switch_persona(self, conversation_id: str, persona_name: str) -> bool:
        """Switch persona for a specific conversation"""
        if persona_name in self._personas:
            self._conversation_personas[conversation_id] = persona_name
            self.logger.info(f"Switched conversation {conversation_id} to persona: {persona_name}")
            return True
        return False
    
    def get_conversation_persona(self, conversation_id: str) -> str:
        """Get current persona for a conversation, default to friendly"""
        return self._conversation_personas.get(conversation_id, "friendly")
    
    async def detect_persona_from_context(self, message: str) -> Optional[str]:
        """Detect appropriate persona based on message content"""
        message_lower = message.lower()
        
        # Check for explicit persona switching
        if message_lower.startswith('/persona'):
            return None  # Let the calling code handle explicit switches
        
        # Technical indicators
        technical_keywords = [
            'yield', 'liquidity', 'apy', 'tvl', 'impermanent loss', 'slippage',
            'smart contract', 'protocol', 'defi', 'amm', 'dex', 'farming',
            'staking', 'governance', 'dao', 'tokenomics', 'audit'
        ]
        
        # Professional/formal indicators
        professional_keywords = [
            'allocation', 'diversification', 'risk management', 'investment',
            'portfolio optimization', 'rebalancing', 'asset management',
            'financial planning', 'wealth', 'advisory'
        ]
        
        # Educational indicators
        educational_keywords = [
            'what is', 'how does', 'can you explain', 'i don\'t understand',
            'new to crypto', 'beginner', 'learn', 'tutorial', 'help me understand'
        ]
        
        # Count keyword matches
        technical_score = sum(1 for keyword in technical_keywords if keyword in message_lower)
        professional_score = sum(1 for keyword in professional_keywords if keyword in message_lower)
        educational_score = sum(1 for keyword in educational_keywords if keyword in message_lower)
        
        # Determine best persona based on scores
        scores = {
            'technical': technical_score,
            'professional': professional_score,
            'educational': educational_score
        }
        
        max_score = max(scores.values())
        if max_score >= 2:  # Minimum threshold for automatic switching
            detected_persona = max(scores, key=scores.get)
            self.logger.info(f"Auto-detected persona: {detected_persona} (score: {max_score})")
            return detected_persona
            
        return None  # Default to current conversation persona
    
    def add_custom_persona(self, persona: Persona) -> bool:
        """Add a custom persona (for future extensibility)"""
        try:
            self._personas[persona.name] = persona
            self.logger.info(f"Added custom persona: {persona.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add custom persona: {str(e)}")
            return False
    
    def get_persona_help_text(self) -> str:
        """Generate help text for persona commands"""
        help_lines = ["Available personas:"]
        
        for name, persona in self._personas.items():
            help_lines.append(f"  • `/persona {name}` - {persona.display_name}: {persona.description}")
        
        help_lines.append("\nExample: `/persona technical` to switch to technical analysis mode")
        
        return "\n".join(help_lines)

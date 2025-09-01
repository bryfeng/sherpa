"""
Persona Management System

This module defines different AI personalities for the agent system,
allowing dynamic switching between communication styles and specializations.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from enum import Enum


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
        """Initialize the default persona set"""
        
        # 1. Friendly Crypto Guide (Default)
        friendly_persona = Persona(
            name="friendly",
            display_name="Friendly Crypto Guide",
            description="Approachable and encouraging crypto assistant",
            system_prompt="""You are a friendly and knowledgeable crypto portfolio assistant named Claude. Your personality is warm, approachable, and encouraging.

COMMUNICATION STYLE:
- Use a conversational, casual tone
- Be encouraging and supportive
- Explain complex concepts using simple analogies
- Show genuine enthusiasm for helping users understand crypto
- Use occasional emojis to add warmth (but don't overdo it)

EXPERTISE:
- Deep knowledge of cryptocurrencies and DeFi
- Portfolio analysis and interpretation
- Market trends and token insights
- Risk assessment and diversification strategies

APPROACH:
- Break down complex information into digestible pieces
- Ask clarifying questions when needed
- Provide context for why certain information matters
- Offer actionable insights and suggestions
- Celebrate portfolio wins and provide comfort during losses

When analyzing portfolios:
- Start with an overview of total value and token count
- Highlight interesting or valuable holdings
- Explain what tokens are and their purposes when relevant
- Point out portfolio diversification strengths or areas for improvement
- Use relatable analogies to explain DeFi concepts

Always maintain a helpful, patient, and encouraging demeanor. You're here to make crypto accessible and less intimidating for everyone.""",
            
            tone="warm and conversational",
            formality="casual",
            technical_depth="medium - explains concepts clearly",
            use_emojis=True,
            response_length="medium",
            specializations=["portfolio_analysis", "crypto_education", "user_encouragement"]
        )
        
        # 2. Technical DeFi Analyst
        technical_persona = Persona(
            name="technical",
            display_name="Technical DeFi Analyst", 
            description="Deep technical knowledge with data-driven insights",
            system_prompt="""You are a highly technical DeFi analyst with deep expertise in blockchain protocols, tokenomics, and quantitative analysis.

COMMUNICATION STYLE:
- Precise, technical language
- Data-driven insights and analysis
- Detailed explanations of protocols and mechanisms
- Focus on metrics, yield strategies, and risk assessment
- Professional but not overly formal

EXPERTISE:
- Advanced DeFi protocol knowledge (Uniswap, Aave, Compound, etc.)
- Yield farming and liquidity mining strategies
- Smart contract risk assessment
- Tokenomics and token utility analysis
- On-chain analytics and metrics interpretation
- MEV, arbitrage, and advanced trading concepts

APPROACH:
- Provide detailed technical breakdowns
- Analyze protocol risks and reward mechanisms
- Explain yield optimization strategies
- Discuss impermanent loss, slippage, and advanced concepts
- Reference specific protocols, APYs, and quantitative metrics
- Address smart contract risks and audit considerations

When analyzing portfolios:
- Calculate risk-adjusted returns and diversification metrics
- Identify yield-generating opportunities
- Assess protocol risks and smart contract exposures
- Analyze token correlations and portfolio efficiency
- Suggest advanced DeFi strategies and optimizations
- Reference specific protocol mechanics and yield farming opportunities

Maintain technical accuracy while ensuring insights are actionable for experienced DeFi users.""",
            
            tone="analytical and precise",
            formality="professional",
            technical_depth="high - detailed technical analysis",
            use_emojis=False,
            response_length="long",
            specializations=["defi_protocols", "yield_strategies", "risk_analysis", "tokenomics"]
        )
        
        # 3. Professional Portfolio Advisor
        professional_persona = Persona(
            name="professional",
            display_name="Professional Portfolio Advisor",
            description="Formal financial guidance with risk-aware recommendations",
            system_prompt="""You are a professional cryptocurrency portfolio advisor with a background in traditional finance and digital asset management.

COMMUNICATION STYLE:
- Formal, structured, and professional
- Clear recommendations with supporting rationale
- Risk-aware and compliance-conscious language
- Focus on portfolio construction and risk management
- Maintain fiduciary-like responsibility in tone

EXPERTISE:
- Portfolio theory applied to cryptocurrency markets
- Risk management and diversification strategies
- Regulatory considerations and compliance
- Traditional finance principles in crypto context
- Asset allocation and rebalancing strategies
- Market analysis and trend identification

APPROACH:
- Provide structured portfolio assessments
- Emphasize risk management and diversification
- Offer specific allocation recommendations
- Discuss correlation analysis and portfolio efficiency
- Address regulatory and tax considerations
- Focus on long-term wealth preservation and growth

When analyzing portfolios:
- Conduct formal portfolio assessment with clear metrics
- Evaluate risk-return profiles and diversification
- Provide specific allocation recommendations
- Identify concentration risks and suggest improvements
- Discuss dollar-cost averaging and rebalancing strategies
- Address tax optimization and regulatory considerations

IMPORTANT DISCLAIMERS:
- Always include appropriate risk disclaimers
- Note that crypto markets are highly volatile and speculative
- Recommend users consult with licensed financial advisors
- Emphasize the importance of only investing what one can afford to lose

Maintain professional standards while providing valuable portfolio guidance.""",
            
            tone="professional and authoritative",
            formality="formal",
            technical_depth="medium-high with focus on portfolio theory",
            use_emojis=False,
            response_length="long",
            specializations=["portfolio_management", "risk_assessment", "financial_planning", "compliance"]
        )
        
        # 4. Educational Crypto Teacher
        educational_persona = Persona(
            name="educational",
            display_name="Educational Crypto Teacher",
            description="Patient teacher focused on crypto education and learning",
            system_prompt="""You are an educational crypto teacher whose primary goal is to help users learn and understand cryptocurrency concepts step by step.

COMMUNICATION STYLE:
- Patient, clear, and methodical
- Break complex topics into learning modules
- Use analogies and real-world examples
- Encourage questions and deeper exploration
- Celebrate learning progress and "aha moments"

EXPERTISE:
- Fundamental blockchain and crypto concepts
- Step-by-step explanations of DeFi mechanisms
- Historical context and market evolution
- Practical tutorials and how-to guidance
- Common mistakes and how to avoid them

APPROACH:
- Start with fundamentals before advanced concepts
- Use the "explain it like I'm 5" methodology when helpful
- Provide context for why concepts matter
- Offer follow-up questions to deepen understanding
- Create learning pathways for different experience levels
- Connect new concepts to previously learned material

When analyzing portfolios:
- Use portfolio as a teaching opportunity
- Explain what each token/protocol does and why it matters
- Discuss the story behind holdings and their purposes
- Identify learning opportunities in the portfolio composition
- Suggest educational resources for deeper learning
- Frame analysis as lessons in portfolio construction

TEACHING TECHNIQUES:
- Use analogies (blockchain = digital ledger, like a bank statement)
- Provide definitions for technical terms
- Offer "quick recap" sections to reinforce learning
- Suggest "next steps" for continued education
- Connect concepts to real-world applications

Always prioritize understanding over complexity, and make crypto education accessible to all experience levels.""",
            
            tone="patient and encouraging",
            formality="friendly but informative",
            technical_depth="adaptive - scales to user's level",
            use_emojis=True,
            response_length="medium-long",
            specializations=["crypto_education", "teaching", "fundamentals", "learning_guidance"]
        )
        
        # Register all personas
        self._personas = {
            "friendly": friendly_persona,
            "technical": technical_persona, 
            "professional": professional_persona,
            "educational": educational_persona
        }
    
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

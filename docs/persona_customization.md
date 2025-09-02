# Persona Customization Guide

This guide explains how to customize AI personas in the Agentic Wallet system. Personas are now stored in user-friendly YAML files that can be easily edited by non-technical users.

## Overview

The Agentic Wallet system uses four distinct AI personas:
- **Friendly Crypto Guide** - Warm, approachable, and encouraging
- **Technical DeFi Analyst** - Deep technical analysis and data-driven insights
- **Professional Portfolio Advisor** - Formal financial guidance with risk awareness
- **Educational Crypto Teacher** - Patient teaching focused on learning

All persona configurations are stored in the `personas/` directory as YAML files.

## File Structure

```
personas/
├── friendly.yaml      # Friendly Crypto Guide configuration
├── technical.yaml     # Technical DeFi Analyst configuration
├── professional.yaml  # Professional Portfolio Advisor configuration
└── educational.yaml   # Educational Crypto Teacher configuration
```

## YAML File Format

Each persona file contains the following sections:

### Basic Information
```yaml
name: friendly                                    # Internal identifier (don't change)
display_name: "Friendly Crypto Guide"           # Name shown to users
description: "Approachable and encouraging crypto assistant"  # Brief description
```

### Communication Style
```yaml
tone: "warm and conversational"                  # Overall communication tone
formality: "casual"                             # Level of formality
technical_depth: "medium - explains concepts clearly"  # How technical responses should be
use_emojis: true                               # Whether to use emojis
response_length: "medium"                       # Preferred response length
```

### Areas of Expertise
```yaml
specializations:                                # Areas this persona excels in
  - portfolio_analysis
  - crypto_education
  - user_encouragement
```

### System Prompt
```yaml
system_prompt: |                               # Core instructions for AI behavior
  You are a friendly and knowledgeable crypto portfolio assistant...
  [Detailed behavior instructions]
```

### Auto-Detection Keywords
```yaml
auto_detection_keywords:                        # Keywords that trigger this persona
  - friendly
  - help
  - explain
  - new to crypto
  - beginner
```

## How to Customize Personas

### 1. Simple Text Changes

To change how a persona introduces itself:
1. Open the relevant YAML file (e.g., `personas/friendly.yaml`)
2. Modify the `display_name` or `description` fields
3. Save the file
4. The changes take effect immediately (no restart required)

**Example:**
```yaml
# Before
display_name: "Friendly Crypto Guide"

# After  
display_name: "Super Helpful Crypto Assistant"
```

### 2. Personality Adjustments

To make a persona more or less formal:
1. Modify the `tone` and `formality` fields
2. Adjust the `system_prompt` to match the new personality
3. Update `use_emojis` if appropriate

**Example - Making the friendly persona more enthusiastic:**
```yaml
tone: "extremely enthusiastic and upbeat"
formality: "very casual and energetic"
use_emojis: true
```

### 3. Technical Depth Changes

To adjust how technical the responses are:
1. Modify the `technical_depth` field
2. Update the system prompt to reflect the new level
3. Adjust specializations if needed

**Options for technical_depth:**
- `"beginner-friendly"` - Simple explanations, lots of analogies
- `"medium"` - Balanced technical content with clear explanations
- `"high"` - Advanced technical analysis and terminology
- `"adaptive"` - Scales complexity based on user's apparent knowledge level

### 4. Response Style Changes

To change response length and structure:
```yaml
response_length: "long"        # Options: short, medium, long, detailed
use_emojis: false             # true/false
```

### 5. Auto-Detection Customization

To change when personas are automatically selected:
1. Add or remove keywords from `auto_detection_keywords`
2. Keywords are matched case-insensitively in user messages

**Example - Adding keywords for technical persona:**
```yaml
auto_detection_keywords:
  - technical
  - analysis  
  - metrics
  - yield farming    # New keyword
  - liquidity pools  # New keyword
  - smart contracts  # New keyword
```

## Advanced Customization

### Custom System Prompts

The `system_prompt` is the most powerful customization option. This text directly instructs the AI on how to behave. You can:

1. **Change personality traits:**
```yaml
system_prompt: |
  You are an extremely patient and encouraging crypto teacher.
  Always celebrate small wins and provide comfort during market downturns.
```

2. **Add specific behaviors:**
```yaml
system_prompt: |
  Always start responses with "Let me break this down for you..."
  Use sports analogies whenever possible to explain crypto concepts.
```

3. **Modify expertise focus:**
```yaml
system_prompt: |
  You specialize in DeFi yield strategies and risk assessment.
  Always mention potential risks before discussing opportunities.
```

### Creating Custom Responses

You can add specific response patterns:

```yaml
system_prompt: |
  When analyzing portfolios:
  1. Always start with a positive observation
  2. Explain token purposes in simple terms
  3. Suggest one concrete improvement
  4. End with encouragement
```

## Best Practices

### 1. Make Small Changes
- Test one change at a time
- See how it affects the AI's responses before making more changes

### 2. Keep Backups
- Copy the original file before making major changes
- You can restore it if needed

### 3. Test Your Changes
```bash
cd agentic_wallet_py
python test_agent_system.py
```

### 4. Be Specific in Instructions
- Vague instructions lead to unpredictable behavior
- Specific behavioral guidelines work better

### 5. Consider Your Audience
- Match the persona to your intended users
- Technical users might prefer less hand-holding
- Beginners need more explanation and encouragement

## Common Customization Examples

### Making a Persona More Conservative
```yaml
system_prompt: |
  Always emphasize risks before opportunities.
  Suggest diversification and conservative strategies.
  Warn about market volatility frequently.
```

### Creating a Fun, Gamified Experience
```yaml
tone: "playful and gamified"
use_emojis: true
system_prompt: |
  Turn crypto education into a game-like experience.
  Use gaming terminology and celebrate "achievements."
  Award imaginary points for learning milestones.
```

### Building Subject Matter Expertise
```yaml
specializations:
  - nft_analysis
  - gaming_tokens
  - metaverse_investments
system_prompt: |
  You are a specialist in NFTs, gaming tokens, and metaverse investments.
  Always relate discussions back to these areas when relevant.
```

## Troubleshooting

### Persona Not Loading
- Check YAML syntax with an online validator
- Ensure all required fields are present
- Check the application logs for error messages

### Changes Not Taking Effect
- Restart the application (personas are loaded at startup)
- Verify the file was saved correctly
- Check file permissions

### Unexpected Behavior
- Review the system prompt for conflicting instructions
- Simplify complex behavioral rules
- Test with different message types

## Technical Notes

- Personas are loaded from `personas/*.yaml` at application startup
- The PersonaManager class handles YAML parsing and validation
- Fallback personas are used if YAML files fail to load
- Auto-detection uses keyword matching with a minimum score threshold

## File Locations

```
agentic_wallet_py/
├── personas/           # Persona configuration files
│   ├── friendly.yaml
│   ├── technical.yaml  
│   ├── professional.yaml
│   └── educational.yaml
├── app/core/agent/
│   └── personas.py     # PersonaManager implementation
└── docs/
    └── persona_customization.md  # This guide
```

---

*For technical implementation details, see the PersonaManager class in `app/core/agent/personas.py`*

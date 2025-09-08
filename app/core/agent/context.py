"""
Context Management System

This module manages conversation history, context compression, and memory
for maintaining coherent conversations across multiple interactions.
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
import json
import uuid
from collections import deque

from ...providers.llm.base import LLMProvider


class ConversationMessage(BaseModel):
    """Individual message in a conversation"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(description="Message role: user, assistant, or system")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    tokens: Optional[int] = Field(default=None, description="Estimated token count for this message")


class ConversationContext(BaseModel):
    """Complete conversation context"""
    
    conversation_id: str
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    total_tokens: int = Field(default=0, description="Total tokens across all messages")
    compressed_history: Optional[str] = Field(default=None, description="Compressed older conversation history")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific preferences")
    portfolio_context: Optional[Dict[str, Any]] = Field(default=None, description="Latest portfolio data for context")
    episodic_focus: Optional[Dict[str, Any]] = Field(default=None, description="Conversation-scoped active focus (entity/metric/timeframe)")


class ContextManager:
    """Manages conversation context, history, and memory"""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        max_tokens: int = 8000,
        compression_threshold: int = 6000,
        max_history_messages: int = 50,
        logger: Optional[logging.Logger] = None
    ):
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.max_history_messages = max_history_messages
        self.logger = logger or logging.getLogger(__name__)
        
        # In-memory storage for conversations
        # In production, this should be replaced with persistent storage
        self._conversations: Dict[str, ConversationContext] = {}
        self._user_preferences: Dict[str, Dict[str, Any]] = {}
    
    async def add_message(
        self,
        conversation_id: str,
        message: Dict[str, Any],
        estimate_tokens: bool = True
    ) -> None:
        """Add a message to conversation history"""
        
        # Get or create conversation context
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )
        
        context = self._conversations[conversation_id]
        
        # Create conversation message
        conv_message = ConversationMessage(
            role=message.get("role", "user"),
            content=message.get("content", ""),
            metadata=message.get("metadata", {})
        )
        
        # Estimate tokens if requested and provider available
        if estimate_tokens and self.llm_provider:
            try:
                conv_message.tokens = self.llm_provider.count_tokens(conv_message.content)
            except Exception as e:
                self.logger.warning(f"Failed to count tokens: {str(e)}")
                conv_message.tokens = len(conv_message.content) // 4  # Rough estimate
        
        # Add message to context
        context.messages.append(conv_message)
        context.last_activity = datetime.now()
        
        if conv_message.tokens:
            context.total_tokens += conv_message.tokens
        
        # Check if context needs compression
        if context.total_tokens > self.compression_threshold:
            await self._compress_context(context)
        
        # Limit message history length
        if len(context.messages) > self.max_history_messages:
            # Remove oldest messages (keep recent ones)
            removed_messages = context.messages[:len(context.messages) - self.max_history_messages]
            context.messages = context.messages[-self.max_history_messages:]
            
            # Update token count
            removed_tokens = sum(msg.tokens or 0 for msg in removed_messages)
            context.total_tokens -= removed_tokens
            
            self.logger.info(f"Trimmed {len(removed_messages)} old messages from conversation {conversation_id}")
    
    async def get_context(self, conversation_id: str, include_portfolio: bool = True) -> str:
        """Get formatted conversation context for LLM"""
        
        if conversation_id not in self._conversations:
            return ""
        
        context = self._conversations[conversation_id]
        context_parts = []
        
        # Add compressed history if available
        if context.compressed_history:
            context_parts.append(f"Previous conversation summary: {context.compressed_history}")
        
        # Add recent message history
        if context.messages:
            recent_messages = context.messages[-10:]  # Last 10 messages for immediate context
            message_history = []
            
            for msg in recent_messages:
                if msg.role in ["user", "assistant"]:
                    message_history.append(f"{msg.role.capitalize()}: {msg.content}")
            
            if message_history:
                context_parts.append("Recent conversation:\n" + "\n".join(message_history))
        
        # Add portfolio context if available and requested
        if include_portfolio and context.portfolio_context:
            portfolio_summary = self._format_portfolio_context(context.portfolio_context)
            context_parts.append(f"Current portfolio context: {portfolio_summary}")

        # Add episodic focus if present
        if context.episodic_focus:
            ef = context.episodic_focus
            try:
                entity = ef.get('protocol') or ef.get('entity') or 'unknown'
                metric = ef.get('metric') or 'metric'
                timeframe = ef.get('window') or ef.get('timeframe') or 'recent'
                chain = ef.get('chain') or 'ethereum'
                summary_bits = []
                stats = ef.get('stats') or {}
                if stats:
                    start_v = stats.get('start_value')
                    end_v = stats.get('end_value')
                    pct = stats.get('pct_change')
                    if start_v is not None and end_v is not None and pct is not None:
                        summary_bits.append(f"start ${start_v:,.0f} → end ${end_v:,.0f} ({pct:+.2f}%)")
                    trend = stats.get('trend')
                    if trend:
                        summary_bits.append(f"trend {trend}")
                summary = ", ".join(summary_bits) if summary_bits else "available"
                context_parts.append(
                    f"Active focus: {entity} {metric} ({timeframe}) on {chain} – {summary}"
                )
            except Exception:
                # Best-effort; ignore formatting issues
                pass
        
        # Add user preferences
        user_id = self._extract_user_id(conversation_id)
        if user_id and user_id in self._user_preferences:
            prefs = self._user_preferences[user_id]
            if prefs:
                context_parts.append(f"User preferences: {json.dumps(prefs, indent=None)}")
        
        return "\n\n".join(context_parts) if context_parts else ""

    async def set_active_focus(self, conversation_id: str, focus: Dict[str, Any]) -> None:
        """Set the conversation's active focus (episodic memory)."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )
        context = self._conversations[conversation_id]
        context.episodic_focus = focus
        context.last_activity = datetime.now()
        self.logger.info(f"Set episodic focus for conversation {conversation_id}: {list(focus.keys())}")

    def get_active_focus(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get the current active focus for the conversation, if any."""
        ctx = self._conversations.get(conversation_id)
        return ctx.episodic_focus if ctx else None
    
    async def integrate_portfolio_data(self, conversation_id: str, portfolio_data: Dict[str, Any]) -> None:
        """Integrate portfolio data into conversation context"""
        
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )
        
        context = self._conversations[conversation_id]
        context.portfolio_context = portfolio_data
        context.last_activity = datetime.now()
        
        self.logger.info(f"Updated portfolio context for conversation {conversation_id}")
    
    async def summarize_old_context(self, conversation_id: str) -> Optional[str]:
        """Summarize older parts of conversation to save tokens"""
        
        if conversation_id not in self._conversations:
            return None
        
        context = self._conversations[conversation_id]
        
        if len(context.messages) < 10:
            return None  # Not enough messages to summarize
        
        # Take messages from the middle of conversation (not too old, not too recent)
        messages_to_summarize = context.messages[:-10]  # All except last 10
        
        if not messages_to_summarize:
            return None
        
        # Create summary prompt
        message_text = []
        for msg in messages_to_summarize:
            if msg.role in ["user", "assistant"]:
                message_text.append(f"{msg.role}: {msg.content}")
        
        conversation_text = "\n".join(message_text)
        
        # If we have an LLM provider, use it to create a summary
        if self.llm_provider:
            try:
                summary = await self._generate_summary_with_llm(conversation_text)
                return summary
            except Exception as e:
                self.logger.warning(f"Failed to generate LLM summary: {str(e)}")
        
        # Fallback to simple summary
        return self._generate_simple_summary(messages_to_summarize)
    
    def get_conversation_stats(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics about a conversation"""
        
        if conversation_id not in self._conversations:
            return None
        
        context = self._conversations[conversation_id]
        
        user_messages = [msg for msg in context.messages if msg.role == "user"]
        assistant_messages = [msg for msg in context.messages if msg.role == "assistant"]
        
        return {
            "conversation_id": conversation_id,
            "total_messages": len(context.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_tokens": context.total_tokens,
            "created_at": context.created_at.isoformat(),
            "last_activity": context.last_activity.isoformat(),
            "has_compressed_history": context.compressed_history is not None,
            "has_portfolio_context": context.portfolio_context is not None
        }
    
    def cleanup_old_conversations(self, max_age_days: int = 7) -> int:
        """Clean up old conversations to free memory"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        conversations_to_remove = []
        
        for conv_id, context in self._conversations.items():
            if context.last_activity < cutoff_date:
                conversations_to_remove.append(conv_id)
        
        for conv_id in conversations_to_remove:
            del self._conversations[conv_id]
        
        if conversations_to_remove:
            self.logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
        
        return len(conversations_to_remove)
    
    async def _compress_context(self, context: ConversationContext) -> None:
        """Compress older parts of conversation to reduce token usage"""
        
        if len(context.messages) < 6:  # Not enough messages to compress
            return
        
        # Summarize older messages
        summary = await self.summarize_old_context(context.conversation_id)
        
        if summary:
            # Store the summary
            context.compressed_history = summary
            
            # Remove the summarized messages (keep last 5 for immediate context)
            messages_to_remove = context.messages[:-5]
            context.messages = context.messages[-5:]
            
            # Update token count
            removed_tokens = sum(msg.tokens or 0 for msg in messages_to_remove)
            context.total_tokens -= removed_tokens
            
            # Add summary token count (rough estimate)
            if self.llm_provider:
                try:
                    summary_tokens = self.llm_provider.count_tokens(summary)
                    context.total_tokens += summary_tokens
                except:
                    context.total_tokens += len(summary) // 4
            
            self.logger.info(f"Compressed context for conversation {context.conversation_id}")
    
    async def _generate_summary_with_llm(self, conversation_text: str) -> str:
        """Generate conversation summary using LLM"""
        
        from ...providers.llm.base import LLMMessage
        
        summary_prompt = f"""Please summarize the following conversation, focusing on:
1. Key topics discussed
2. Important portfolio or crypto information mentioned
3. User preferences or recurring themes
4. Any decisions or conclusions reached

Conversation:
{conversation_text}

Provide a concise but comprehensive summary in 2-3 sentences:"""
        
        messages = [LLMMessage(role="user", content=summary_prompt)]
        
        response = await self.llm_provider.generate_response(
            messages=messages,
            max_tokens=200,
            temperature=0.3
        )
        
        return response.content.strip()
    
    def _generate_simple_summary(self, messages: List[ConversationMessage]) -> str:
        """Generate simple summary without LLM"""
        
        # Extract key information
        topics = set()
        portfolio_mentions = 0
        
        for msg in messages:
            content_lower = msg.content.lower()
            
            # Look for topic keywords
            if any(keyword in content_lower for keyword in ["portfolio", "wallet", "balance"]):
                portfolio_mentions += 1
                topics.add("portfolio analysis")
            
            if any(keyword in content_lower for keyword in ["defi", "yield", "farming"]):
                topics.add("DeFi discussion")
            
            if any(keyword in content_lower for keyword in ["token", "crypto", "price"]):
                topics.add("cryptocurrency topics")
        
        summary_parts = []
        
        if topics:
            summary_parts.append(f"Discussed: {', '.join(topics)}")
        
        if portfolio_mentions > 0:
            summary_parts.append(f"Portfolio mentioned {portfolio_mentions} times")
        
        summary_parts.append(f"({len(messages)} messages exchanged)")
        
        return " | ".join(summary_parts)
    
    def _format_portfolio_context(self, portfolio_data: Dict[str, Any]) -> str:
        """Format portfolio data for context inclusion"""
        
        try:
            address = portfolio_data.get('address', 'Unknown')
            total_value = portfolio_data.get('total_value_usd', '0')
            token_count = portfolio_data.get('token_count', 0)
            
            return f"Wallet {address} with ${total_value} USD across {token_count} tokens"
        except:
            return "Portfolio data available"
    
    def _extract_user_id(self, conversation_id: str) -> Optional[str]:
        """Extract user ID from conversation ID (placeholder implementation)"""
        # In a real implementation, this would extract user ID from conversation ID
        # For now, return conversation_id as a simple user identifier
        return conversation_id.split('-')[0] if '-' in conversation_id else conversation_id


class ConversationMemory:
    """Handles persistent conversation memory and user preferences"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # In-memory storage - in production, use database
        self._conversation_store: Dict[str, Dict[str, Any]] = {}
        self._user_preferences: Dict[str, Dict[str, Any]] = {}
    
    def store_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store conversation messages"""
        
        self._conversation_store[conversation_id] = {
            'messages': messages,
            'stored_at': datetime.now().isoformat(),
            'message_count': len(messages)
        }
        
        self.logger.debug(f"Stored {len(messages)} messages for conversation {conversation_id}")
    
    def retrieve_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation messages"""
        
        conversation_data = self._conversation_store.get(conversation_id, {})
        return conversation_data.get('messages', [])
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        
        return self._user_preferences.get(user_id, {})
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Update user preferences"""
        
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        
        self._user_preferences[user_id].update(preferences)
        
        self.logger.info(f"Updated preferences for user {user_id}: {list(preferences.keys())}")
    
    def get_conversation_list(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of conversations, optionally filtered by user"""
        
        conversations = []
        
        for conv_id, data in self._conversation_store.items():
            conversation_info = {
                'conversation_id': conv_id,
                'message_count': data.get('message_count', 0),
                'stored_at': data.get('stored_at'),
            }
            
            # Simple user filtering (in production, use proper user associations)
            if user_id is None or conv_id.startswith(user_id):
                conversations.append(conversation_info)
        
        return sorted(conversations, key=lambda x: x['stored_at'] or '', reverse=True)
    
    def cleanup_old_data(self, max_age_days: int = 30) -> Tuple[int, int]:
        """Clean up old conversation data and inactive user preferences"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean up old conversations
        conversations_removed = 0
        conversations_to_remove = []
        
        for conv_id, data in self._conversation_store.items():
            try:
                stored_at = datetime.fromisoformat(data.get('stored_at', ''))
                if stored_at < cutoff_date:
                    conversations_to_remove.append(conv_id)
            except:
                # If we can't parse the date, consider it old
                conversations_to_remove.append(conv_id)
        
        for conv_id in conversations_to_remove:
            del self._conversation_store[conv_id]
            conversations_removed += 1
        
        # Note: We don't automatically clean user preferences as they may be valuable long-term
        
        if conversations_removed > 0:
            self.logger.info(f"Cleaned up {conversations_removed} old conversations")
        
        return conversations_removed, 0

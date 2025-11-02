"""
Advanced OpenAI-based Summarization Module
This module provides sophisticated text summarization using OpenAI API directly
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
import logging

# OpenAI imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"OpenAI import failed: {e}")
    OPENAI_AVAILABLE = False
    openai = None
    OpenAI = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationSummarizer:
    """
    Advanced notification summarizer using OpenAI API directly
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the summarizer with OpenAI API key
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. Summarizer will use fallback method.")
            self.client = None
        else:
            if OPENAI_AVAILABLE:
                self.client = OpenAI(api_key=self.api_key)
            else:
                logger.warning("OpenAI package not available. Using fallback.")
                self.client = None
    
    def _call_openai(self, messages: list, max_tokens: int = 150) -> str:
        """Call OpenAI API with messages"""
        if not self.client:
            raise Exception("OpenAI client not available")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise e
    
    def summarize_basic(self, text: str) -> str:
        """
        Basic summarization using simple prompt
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary string
        """
        if not self.client:
            return self._fallback_summary(text)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at summarizing notifications concisely. Create brief, clear summaries that capture the essential information. Keep summaries under 50 words and focus on actionable information."
                },
                {
                    "role": "user", 
                    "content": f"Summarize this notification: {text}"
                }
            ]
            
            result = self._call_openai(messages)
            logger.info("Basic summary generated successfully")
            return result
                
        except Exception as e:
            logger.error(f"Error in basic summarization: {str(e)}")
            return self._fallback_summary(text)
    
    def summarize_contextual(self, package_name: str, title: str, content: str) -> Dict[str, Any]:
        """
        Advanced contextual summarization with app-specific insights
        
        Args:
            package_name: App package name
            title: Notification title
            content: Notification content
            
        Returns:
            Dictionary with summary and metadata
        """
        if not self.client:
            return {
                "summary": self._fallback_summary(content),
                "app_type": self._get_app_type(package_name),
                "confidence": "low"
            }
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a smart notification summarizer. Based on the app type and content, create relevant summaries:
                    - For messaging apps: Who messaged and brief content
                    - For emails: Sender and subject/key points
                    - For social media: Platform and type of notification
                    - For system apps: What action or update occurred
                    - For news apps: Headline and key information
                    
                    Keep summaries under 50 words and highlight important details."""
                },
                {
                    "role": "user",
                    "content": f"""
                    App: {package_name}
                    Title: {title}
                    Content: {content}
                    """
                }
            ]
            
            result = self._call_openai(messages)
            logger.info("Contextual summary generated successfully")
            
            return {
                "summary": result,
                "app_type": self._get_app_type(package_name),
                "confidence": "high",
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Error in contextual summarization: {str(e)}")
            return {
                "summary": self._fallback_summary(content),
                "app_type": self._get_app_type(package_name),
                "confidence": "low",
                "error": str(e)
            }
    
    def summarize_with_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Advanced summarization with sentiment analysis and urgency detection
        
        Args:
            text: Text to analyze and summarize
            
        Returns:
            Dictionary with summary, sentiment, and urgency
        """
        if not self.client:
            fallback = self._fallback_summary(text)
            return {
                "summary": fallback,
                "urgency": "medium",
                "sentiment": "neutral",
                "confidence": "low"
            }
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a notification summarizer that includes sentiment analysis.
                    Create a summary that includes:
                    1. Main message (under 40 words)
                    2. Urgency level (Low/Medium/High)
                    3. Sentiment (Positive/Neutral/Negative)
                    
                    Format: "Summary | Urgency: [level] | Sentiment: [sentiment]" """
                },
                {
                    "role": "user",
                    "content": f"Notification: {text}"
                }
            ]
            
            result = self._call_openai(messages)
            
            # Parse the structured response
            parsed = self._parse_sentiment_response(result)
            parsed["timestamp"] = datetime.now().isoformat()
            
            logger.info("Sentiment summary generated successfully")
            return parsed
                
        except Exception as e:
            logger.error(f"Error in sentiment summarization: {str(e)}")
            return {
                "summary": self._fallback_summary(text),
                "urgency": "medium",
                "sentiment": "neutral",
                "confidence": "low",
                "error": str(e)
            }
    
    def smart_summarize(self, package_name: str, title: str, content: str, raw_text: str) -> Dict[str, Any]:
        """
        Intelligent summarization that chooses the best strategy based on content
        
        Args:
            package_name: App package name
            title: Notification title
            content: Notification content
            raw_text: Raw notification text
            
        Returns:
            Comprehensive summary with metadata
        """
        # Determine best summarization strategy
        text_length = len(content)
        app_type = self._get_app_type(package_name)
        
        # Choose strategy based on content characteristics
        if text_length > 200 or app_type in ["email", "news", "social"]:
            # Use contextual summarization for complex content
            result = self.summarize_contextual(package_name, title, content)
            result["strategy"] = "contextual"
        elif self._contains_sentiment_indicators(content):
            # Use sentiment analysis for emotional content
            result = self.summarize_with_sentiment(content)
            result["strategy"] = "sentiment"
        else:
            # Use basic summarization for simple content
            summary = self.summarize_basic(content)
            result = {
                "summary": summary,
                "strategy": "basic",
                "app_type": app_type,
                "confidence": "medium"
            }
        
        # Add common metadata
        result.update({
            "original_length": text_length,
            "compression_ratio": len(result.get("summary", "")) / max(text_length, 1),
            "processed_at": datetime.now().isoformat()
        })
        
        return result
    
    def _fallback_summary(self, text: str) -> str:
        """Fallback summarization when LLM is not available"""
        if len(text) <= 100:
            return text
        
        # Simple extractive summarization
        sentences = text.split('. ')
        if len(sentences) <= 2:
            return text[:100] + "..." if len(text) > 100 else text
        
        # Take first and most important sentence
        important_words = ['urgent', 'important', 'update', 'new', 'alert', 'reminder']
        important_sentence = None
        
        for sentence in sentences:
            if any(word.lower() in sentence.lower() for word in important_words):
                important_sentence = sentence
                break
        
        if important_sentence:
            return f"{sentences[0]}. {important_sentence}."
        else:
            return sentences[0] + (f". {sentences[1]}." if len(sentences) > 1 else "")
    
    def _get_app_type(self, package_name: str) -> str:
        """Determine app type from package name"""
        app_types = {
            "messaging": ["whatsapp", "telegram", "messages", "sms", "messenger", "slack", "discord"],
            "email": ["gmail", "outlook", "mail", "email"],
            "social": ["facebook", "instagram", "twitter", "linkedin", "snapchat", "tiktok"],
            "news": ["news", "bbc", "cnn", "reuters", "medium"],
            "productivity": ["calendar", "reminder", "notes", "drive", "dropbox"],
            "entertainment": ["youtube", "spotify", "netflix", "music"],
            "system": ["android", "system", "settings", "security"]
        }
        
        package_lower = package_name.lower()
        for app_type, keywords in app_types.items():
            if any(keyword in package_lower for keyword in keywords):
                return app_type
        
        return "other"
    
    def _contains_sentiment_indicators(self, text: str) -> bool:
        """Check if text contains sentiment indicators"""
        sentiment_words = [
            'urgent', 'important', 'critical', 'warning', 'error', 'failed',
            'congratulations', 'success', 'completed', 'approved', 'rejected',
            'love', 'hate', 'angry', 'happy', 'sad', 'excited'
        ]
        
        text_lower = text.lower()
        return any(word in text_lower for word in sentiment_words)
    
    def _parse_sentiment_response(self, response: str) -> Dict[str, Any]:
        """Parse structured sentiment response"""
        try:
            parts = response.split(" | ")
            summary = parts[0]
            urgency = "medium"
            sentiment = "neutral"
            
            for part in parts[1:]:
                if part.startswith("Urgency:"):
                    urgency = part.split(":")[1].strip().lower()
                elif part.startswith("Sentiment:"):
                    sentiment = part.split(":")[1].strip().lower()
            
            return {
                "summary": summary,
                "urgency": urgency,
                "sentiment": sentiment,
                "confidence": "high"
            }
        except Exception:
            return {
                "summary": response,
                "urgency": "medium",
                "sentiment": "neutral",
                "confidence": "low"
            }

# Global instance
_summarizer = None

def get_summarizer() -> NotificationSummarizer:
    """Get global summarizer instance"""
    global _summarizer
    if _summarizer is None:
        _summarizer = NotificationSummarizer()
    return _summarizer

def summarize_notification(package_name: str, title: str, content: str, raw_text: str) -> Dict[str, Any]:
    """
    Main function to summarize notifications
    
    Args:
        package_name: App package name
        title: Notification title
        content: Notification content
        raw_text: Raw notification text
        
    Returns:
        Summary dictionary
    """
    summarizer = get_summarizer()
    return summarizer.smart_summarize(package_name, title, content, raw_text)
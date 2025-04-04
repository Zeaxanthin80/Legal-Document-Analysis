from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# LLMChain is deprecated, we'll use LCEL (prompt | llm)
# from langchain.chains import LLMChain 
import spacy
import logging
from collections import Counter
import json # Import json module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalClassifier:
    """A class to analyze and classify legal documents."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the LegalClassifier.
        
        Args:
            model_name (str): Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(model_name=model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal document classifier. Analyze the following text and classify it into relevant legal categories. Respond with the classification only."),
            ("human", "{text}")
        ])
        # self.classification_chain = LLMChain(...) # Deprecated
        self.classification_chain = self.classification_prompt | self.llm
        
        # Define topic analysis prompt
        self.topic_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the following legal text and identify the main legal topics. Return ONLY a valid JSON object (no other text) with a single key 'topics' mapping to a dictionary of topic names and their relevance scores (0-1)."),
            ("human", "{text}")
        ])
        # topic_chain = LLMChain(...) # Deprecated
        self.topic_chain = self.topic_prompt | self.llm
        
        # Define summarization prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following legal document, focusing on key points and legal implications."),
            ("human", "{text}")
        ])
        # summary_chain = LLMChain(...) # Deprecated
        self.summary_chain = self.summary_prompt | self.llm
        
    def classify_document(self, document: Document) -> Dict[str, Any]:
        """
        Classify a legal document into relevant categories.
        
        Args:
            document (Document): Document to classify
            
        Returns:
            Dict[str, Any]: Classification results
        """
        try:
            # Get classification from LLM using .invoke()
            # result = self.classification_chain.run(text=document.page_content) # Deprecated
            result = self.classification_chain.invoke({"text": document.page_content})
            classification = result.content # Assuming result is an AIMessage
            
            # Extract entities using spaCy
            doc = self.nlp(document.page_content)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Count entity types
            entity_counts = Counter(label for _, label in entities)
            
            return {
                "classification": classification,
                "entities": entities,
                "entity_counts": dict(entity_counts),
                "metadata": document.metadata
            }
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return {}
            
    def extract_legal_entities(self, document: Document) -> List[Dict[str, str]]:
        """
        Extract legal entities from a document.
        
        Args:
            document (Document): Document to analyze
            
        Returns:
            List[Dict[str, str]]: List of extracted entities
        """
        try:
            doc = self.nlp(document.page_content)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LAW"]:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
                    
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
            
    def analyze_legal_topics(self, document: Document) -> Dict[str, float]:
        """
        Analyze the main legal topics in a document.
        
        Args:
            document (Document): Document to analyze
            
        Returns:
            Dict[str, float]: Topic scores dictionary, or empty dict on error.
        """
        try:
            # result = topic_chain.run(text=document.page_content) # Deprecated
            result = self.topic_chain.invoke({"text": document.page_content})
            json_string = result.content # Assuming result is an AIMessage
            
            # Parse the JSON string output from LLM
            try:
                topics_data = json.loads(json_string)
                # Validate structure slightly
                if isinstance(topics_data, dict) and 'topics' in topics_data and isinstance(topics_data['topics'], dict):
                    return topics_data['topics'] # Return the inner dictionary
                else:
                    logger.warning(f"LLM returned unexpected JSON structure: {json_string}")
                    return {}
            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to decode JSON from LLM: {json_e}")
                logger.error(f"LLM output was: {json_string}")
                return {}
            
        except Exception as e:
            logger.error(f"Error analyzing topics: {str(e)}")
            return {}
            
    def summarize_document(self, document: Document) -> str:
        """
        Generate a summary of the legal document.
        
        Args:
            document (Document): Document to summarize
            
        Returns:
            str: Document summary
        """
        try:
            # result = summary_chain.run(text=document.page_content) # Deprecated
            result = self.summary_chain.invoke({"text": document.page_content})
            summary = result.content # Assuming result is an AIMessage
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            return ""
            
    def analyze_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of documents.
        
        Args:
            documents (List[Document]): List of documents to analyze
            
        Returns:
            List[Dict[str, Any]]: Analysis results for each document
        """
        results = []
        for doc in documents:
            # Get the full classification result dictionary
            classification_result = self.classify_document(doc)
            # Get the topics dictionary directly
            topics_dict = self.analyze_legal_topics(doc)
            
            analysis = {
                # classification_result already contains metadata, entities etc.
                "classification_data": classification_result, 
                "topics": topics_dict, # Use the parsed dictionary
                "summary": self.summarize_document(doc)
                # No need to call extract_legal_entities separately if included in classify_document
            }
            results.append(analysis)
        return results 
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import spacy
import logging
from collections import Counter

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
            ("system", "You are a legal document classifier. Analyze the following text and classify it into relevant legal categories."),
            ("human", "{text}")
        ])
        
        self.classification_chain = LLMChain(
            llm=self.llm,
            prompt=self.classification_prompt
        )
        
    def classify_document(self, document: Document) -> Dict[str, Any]:
        """
        Classify a legal document into relevant categories.
        
        Args:
            document (Document): Document to classify
            
        Returns:
            Dict[str, Any]: Classification results
        """
        try:
            # Get classification from LLM
            classification = self.classification_chain.run(text=document.page_content)
            
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
            Dict[str, float]: Topic scores
        """
        try:
            # Define topic analysis prompt
            topic_prompt = ChatPromptTemplate.from_messages([
                ("system", "Analyze the following legal text and identify the main legal topics. Return a JSON object with topics and their relevance scores (0-1)."),
                ("human", "{text}")
            ])
            
            topic_chain = LLMChain(llm=self.llm, prompt=topic_prompt)
            topics = topic_chain.run(text=document.page_content)
            
            return topics
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
            # Define summarization prompt
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the following legal document, focusing on key points and legal implications."),
                ("human", "{text}")
            ])
            
            summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
            summary = summary_chain.run(text=document.page_content)
            
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
            analysis = {
                "classification": self.classify_document(doc),
                "entities": self.extract_legal_entities(doc),
                "topics": self.analyze_legal_topics(doc),
                "summary": self.summarize_document(doc)
            }
            results.append(analysis)
        return results 
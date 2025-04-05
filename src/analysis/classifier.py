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
# Import the output parser
from langchain_core.output_parsers import JsonOutputParser

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
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Instantiate the output parser
        self.json_parser = JsonOutputParser()
        
        # --- Refined Prompts --- 

        # Classification Prompt (more specific instructions)
        classification_system_message = (
            "You are an expert legal assistant specializing in Florida Statutes. "
            "Your task is to classify the provided legal text snippet into one or more relevant categories. "
            "Focus on the primary legal subject matter. Examples of categories include: "
            "Juvenile Justice, Family Law, Tort Law, Criminal Procedure, Dependency Proceedings, Civil Procedure, Property Law. "
            "Respond with a comma-separated list of the most relevant categories ONLY. Do not add explanation.\n"
            "\nExample:\n"
            "Text: 'The state and its agencies and subdivisions shall be liable for tort claims...'\n"
            "Response: Tort Law, Sovereign Immunity"
        )
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", classification_system_message),
            ("human", "Classify the following legal text: \n\n{text}")
        ])
        self.classification_chain = self.classification_prompt | self.llm
        
        # Topic Analysis Prompt (using JsonOutputParser instructions)
        topic_system_message = (
            "You are an expert legal analyst. Analyze the provided legal text snippet and identify the main legal topics discussed. "
            "Focus on specific legal concepts or areas mentioned. "
            "Return ONLY the JSON object containing the topics and their confidence scores (0.0-1.0)."
            "\n{format_instructions}"
        )
        # Create the template first without partial_variables
        topic_prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                ("system", topic_system_message),
                ("human", "Analyze the topics in the following legal text: \n\n{text}")
            ]
            # No partial_variables here
        )
        # Apply partial variables using the .partial() method
        self.topic_prompt = topic_prompt_template.partial(
            format_instructions=self.json_parser.get_format_instructions()
        )
        # Pipe the LLM output through the JSON parser
        self.topic_chain = self.topic_prompt | self.llm | self.json_parser
        
        # Define summarization prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following legal document snippet concisely, focusing on key legal points, definitions, procedures, or implications."),
            ("human", "{text}")
        ])
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
            
    def analyze_legal_topics(self, document: Document) -> Dict[str, Any]: # Return type might be Any if parsing fails
        """
        Analyze the main legal topics in a document using JsonOutputParser.
        
        Args:
            document (Document): Document to analyze
            
        Returns:
            Dict[str, Any]: Parsed topic dictionary, or empty dict on error.
        """
        try:
            # Invoke the chain - the parser handles JSON validation/parsing
            parsed_output = self.topic_chain.invoke({"text": document.page_content})
            
            # The JsonOutputParser should return a dict directly if successful
            # We might still want basic structure validation depending on strictness needed
            if isinstance(parsed_output, dict):
                 # We might need to adapt this if the parser expects a specific structure
                 # For now, let's assume it returns {"topics": {...}} as requested in the old prompt
                 # If JsonOutputParser just returns the direct dict, we adapt.
                 # Let's assume for now it returns the structure we want directly.
                 # Check the actual output format if this fails. 
                 # return parsed_output.get("topics", {}) 
                 return parsed_output # Return the parsed dict directly
            else:
                 logger.warning(f"JsonOutputParser returned non-dict type: {type(parsed_output)}")
                 return {}
                 
        except Exception as e:
            # Errors could be from LLM call OR from the JsonOutputParser failing
            logger.error(f"Error analyzing topics (potentially parsing error): {str(e)}")
            # Attempting to log the raw output might fail if the error was before LLM call
            # try:
            #     raw_llm_output = (self.topic_prompt | self.llm).invoke({"text": document.page_content})
            #     logger.error(f"Raw LLM output before parsing failure: {raw_llm_output.content}")
            # except Exception as inner_e:
            #     logger.error(f"Could not retrieve raw LLM output: {inner_e}")
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
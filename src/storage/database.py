import sqlite3
import json
from datetime import datetime
from pathlib import Path
import hashlib
import logging
from typing import List, Tuple, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the default database file path relative to the project root
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "metadata.db"

class DatabaseManager:
    """Manages interactions with the SQLite database for storing document metadata and analysis results."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        """Initialize the DatabaseManager. Connection is established in connect()."""
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
        # Ensure directory exists on init
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        """Establishes the database connection."""
        if self.conn is None:
            try:
                self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, check_same_thread=False) # Allow cross-thread use IF CAREFUL
                # OR: self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) # Default, safer
                self.cursor = self.conn.cursor()
                logging.debug(f"Connected to database: {self.db_path.resolve()}")
                self._create_tables()
            except sqlite3.Error as e:
                logging.error(f"Database connection error to {self.db_path}: {e}")
                raise

    def _create_tables(self):
        """Creates the necessary tables if they do not already exist."""
        try:
            # documents table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    num_chunks INTEGER,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # analysis_results table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    analysis_type TEXT NOT NULL,
                    result_content TEXT, -- Storing JSON as TEXT
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
                )
            """)

            # chat_history table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL, -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # index for faster session retrieval
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_session_ts ON chat_history (session_id, timestamp);
            """)

            self.conn.commit()
            logging.info("Database tables verified/created successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error creating database tables: {e}")
            self.conn.rollback() # Rollback changes if table creation failed

    def add_document(self, file_path: str, file_hash: str, num_chunks: int) -> int | None:
        """
        Adds a record for a processed document.

        Args:
            file_path: Relative path to the original document file.
            file_hash: SHA256 hash of the document file.
            num_chunks: Number of chunks generated for the document.

        Returns:
            The doc_id of the newly inserted document, or None if insertion failed.
        """
        try:
            self.cursor.execute("""
                INSERT INTO documents (file_path, file_hash, num_chunks, indexed_at)
                VALUES (?, ?, ?, ?)
            """, (file_path, file_hash, num_chunks, datetime.now()))
            self.conn.commit()
            doc_id = self.cursor.lastrowid
            logging.info(f"Added document record: path='{file_path}', hash='{file_hash[:8]}...', chunks={num_chunks}, doc_id={doc_id}")
            return doc_id
        except sqlite3.IntegrityError:
            logging.warning(f"Document with path '{file_path}' likely already exists.")
            self.conn.rollback()
            # Optionally, return the existing doc_id here if needed
            return self.get_doc_id_by_path(file_path)
        except sqlite3.Error as e:
            logging.error(f"Error adding document record for '{file_path}': {e}")
            self.conn.rollback()
            return None

    def check_document_processed(self, file_path: str, file_hash: str) -> bool:
        """
        Checks if a document with the same path and hash already exists in the database.

        Args:
            file_path: Relative path to the document file.
            file_hash: SHA256 hash of the document file.

        Returns:
            True if a matching document record exists, False otherwise.
        """
        try:
            self.cursor.execute("""
                SELECT 1 FROM documents WHERE file_path = ? AND file_hash = ?
            """, (file_path, file_hash))
            result = self.cursor.fetchone()
            processed = result is not None
            if processed:
                logging.debug(f"Document already processed: path='{file_path}', hash='{file_hash[:8]}...'")
            else:
                 logging.debug(f"Document not yet processed or changed: path='{file_path}', hash='{file_hash[:8]}...'")
            return processed
        except sqlite3.Error as e:
            logging.error(f"Error checking document '{file_path}': {e}")
            return False # Assume not processed if there's an error

    def get_doc_id_by_path(self, file_path: str) -> int | None:
        """
        Retrieves the doc_id for a given file path.

        Args:
            file_path: Relative path to the document file.

        Returns:
            The doc_id if found, otherwise None.
        """
        try:
            self.cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (file_path,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logging.error(f"Error retrieving doc_id for '{file_path}': {e}")
            return None

    def add_analysis_result(self, doc_id: int, analysis_type: str, result_content: dict | list) -> int | None:
        """
        Adds a record for an analysis result, storing the content as a JSON string.

        Args:
            doc_id: The ID of the document that was analyzed.
            analysis_type: The type of analysis performed (e.g., 'topic_modeling').
            result_content: The analysis result (dict or list) to be stored as JSON.

        Returns:
            The result_id of the newly inserted analysis result, or None if insertion failed.
        """
        if doc_id is None:
             logging.error(f"Cannot add analysis result: invalid doc_id (None) for type '{analysis_type}'.")
             return None
        try:
            result_json = json.dumps(result_content) # Convert Python object to JSON string
            self.cursor.execute("""
                INSERT INTO analysis_results (doc_id, analysis_type, result_content, analyzed_at)
                VALUES (?, ?, ?, ?)
            """, (doc_id, analysis_type, result_json, datetime.now()))
            self.conn.commit()
            result_id = self.cursor.lastrowid
            logging.info(f"Added analysis result: doc_id={doc_id}, type='{analysis_type}', result_id={result_id}")
            return result_id
        except sqlite3.Error as e:
            logging.error(f"Error adding analysis result for doc_id {doc_id}, type '{analysis_type}': {e}")
            self.conn.rollback()
            return None
        except TypeError as e:
             logging.error(f"Error serializing analysis result to JSON for doc_id {doc_id}, type '{analysis_type}': {e}")
             return None

    def close(self):
        """Closes the database connection if it's open."""
        if self.cursor:
            try:
                self.cursor.close()
            except sqlite3.Error as e:
                 logging.debug(f"Error closing cursor: {e}")
            finally:
                self.cursor = None
        if self.conn:
            try:
                # Commit before close might be needed depending on workflow
                # self.conn.commit() # Be careful with auto-commit
                self.conn.close()
                logging.debug("Database connection closed.")
            except sqlite3.Error as e:
                logging.debug(f"Error closing connection: {e}")
            finally:
                self.conn = None

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

    def __del__(self):
        # Optional: Still useful as a fallback
        # self.close()
        pass # Let __exit__ handle explicit closure

    def get_all_documents(self) -> List[Tuple[int, str]]:
        """Retrieves the ID and path of all documents in the database."""
        query = "SELECT doc_id, file_path FROM documents ORDER BY file_path;"
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results if results else []
        except sqlite3.Error as e:
            logging.error(f"Error fetching all documents: {e}")
            return []

    def get_analysis_types_for_doc(self, doc_id: int) -> List[str]:
        """Retrieves distinct analysis types available for a given document ID."""
        query = "SELECT DISTINCT analysis_type FROM analysis_results WHERE doc_id = ? ORDER BY analysis_type;"
        params = (doc_id,)
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return [row[0] for row in results] if results else []
        except sqlite3.Error as e:
            logging.error(f"Error fetching analysis types for doc_id {doc_id}: {e}")
            return []

    def get_analysis_result(self, doc_id: int, analysis_type: str) -> Optional[str]:
        """Retrieves the stored JSON result for a specific document and analysis type."""
        query = "SELECT result_content FROM analysis_results WHERE doc_id = ? AND analysis_type = ?;"
        params = (doc_id, analysis_type)
        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logging.error(f"Error fetching analysis result for doc_id {doc_id}, type {analysis_type}: {e}")
            return None

    def add_chat_message(self, session_id: str, role: str, content: str) -> Optional[int]:
        """Adds a chat message to the history."""
        query = "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?);"
        params = (session_id, role, content)
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            message_id = self.cursor.lastrowid
            logging.debug(f"Added chat message id={message_id} for session={session_id}")
            return message_id
        except sqlite3.Error as e:
            logging.error(f"Error adding chat message for session {session_id}: {e}")
            self.conn.rollback() # Rollback on error
            return None

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves chat history for a given session ID, ordered by timestamp."""
        query = "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC;"
        params = (session_id,)
        history = []
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            for row in results:
                history.append({"role": row[0], "content": row[1]})
            logging.debug(f"Retrieved {len(history)} messages for session {session_id}")
            return history
        except sqlite3.Error as e:
            logging.error(f"Error fetching chat history for session {session_id}: {e}")
            return [] # Return empty list on error

# Helper function for hashing files
def calculate_file_hash(file_path: str | Path) -> str:
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        logging.error(f"Cannot calculate hash: File not found at {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Error calculating hash for {file_path}: {e}")
        return "" 
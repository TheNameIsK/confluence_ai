import os
import sqlite3
import json
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter, Retry
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from prompt import sysprompt

# Load environment variables
load_dotenv()

# Configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI Configuration
base = os.getenv('OPENAI_BASE_URL')
openai = OpenAI(base_url=base)

# Confluence Configuration
CONFLUENCE_URL = os.getenv('CONFLUENCE_URL')
CONFLUENCE_USERNAME = os.getenv('CONFLUENCE_USERNAME')
CONFLUENCE_API_TOKEN = os.getenv('CONFLUENCE_API_TOKEN')

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass
class Document:
    id: str
    title: str
    content: str
    space_key: str
    space_name: str
    type: str
    url: str
    last_modified: str
    excerpt: str = ""

class DatabaseManager:
    def __init__(self, db_path='confluence_rag.db'):
        self.db_path = db_path
        # Persistent connection with thread safety
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        with self.lock:
            cursor = self.conn.cursor()
            
            # Documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    space_key TEXT NOT NULL,
                    space_name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    url TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    excerpt TEXT,
                    embedding BLOB
                )
            ''')
            
            # Spaces table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spaces (
                    key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    last_synced TEXT
                )
            ''')
            
            # Chat history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # Sync metadata table for tracking incremental changes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            # Full-text search virtual table for titles
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_documents USING fts5(
                    id, title, content, excerpt
                )
            ''')
            
            self.conn.commit()
            
    def get_page_by_title(self, title: str, space_name: str = None) -> dict:
        """
        Find a page by its title, optionally within a specific space.
        
        Args:
            title (str): The title of the page to find
            space_name (str, optional): The space name to search within for disambiguation
        
        Returns:
            dict: Page information including ID, or None if not found
        """
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                if space_name:
                    # Search within specific space
                    cursor.execute('''
                        SELECT id, title, content, space_key, space_name, type, url, last_modified, excerpt
                        FROM documents 
                        WHERE title = ? AND space_name = ?
                        LIMIT 1
                    ''', (title, space_name))
                else:
                    # Search across all spaces
                    cursor.execute('''
                        SELECT id, title, content, space_key, space_name, type, url, last_modified, excerpt
                        FROM documents 
                        WHERE title = ?
                        LIMIT 1
                    ''', (title,))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'space_key': result[3],
                        'space_name': result[4],
                        'type': result[5],
                        'url': result[6],
                        'last_modified': result[7],
                        'excerpt': result[8]
                    }
                else:
                    return None
                    
        except sqlite3.Error as e:
            logger.error(f"Database error in get_page_by_title: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting page by title: {e}")
            return None

    def get_pages_by_title_fuzzy(self, title: str, space_name: str = None, limit: int = 5) -> list:
        """
        Find pages by title with fuzzy matching (case-insensitive, partial matches).
        Useful when exact title match fails.
        
        Args:
            title (str): The title to search for
            space_name (str, optional): The space name to search within
            limit (int): Maximum number of results to return
        
        Returns:
            list: List of matching pages with similarity scores
        """
        try:
            with self.lock:
                cursor = self.conn.cursor()

                query = f"title:{title}*"

                if space_name:
                    cursor.execute('''
                        SELECT d.id, d.title, d.content, d.space_key, d.space_name, d.type,
                               d.url, d.last_modified, d.excerpt
                        FROM documents d
                        JOIN fts_documents f ON d.id = f.id
                        WHERE f MATCH ? AND d.space_name = ?
                        LIMIT ?
                    ''', (query, space_name, limit))
                else:
                    cursor.execute('''
                        SELECT d.id, d.title, d.content, d.space_key, d.space_name, d.type,
                               d.url, d.last_modified, d.excerpt
                        FROM documents d
                        JOIN fts_documents f ON d.id = f.id
                        WHERE f MATCH ?
                        LIMIT ?
                    ''', (query, limit))
                
                results = cursor.fetchall()
                
                pages = []
                for result in results:
                    pages.append({
                        'id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'space_key': result[3],
                        'space_name': result[4],
                        'type': result[5],
                        'url': result[6],
                        'last_modified': result[7],
                        'excerpt': result[8]
                    })
                
                return pages
                
        except sqlite3.Error as e:
            logger.error(f"Database error in get_pages_by_title_fuzzy: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting pages by title fuzzy: {e}")
            return []

    def get_page_by_title_with_suggestions(self, title: str, space_name: str = None) -> dict:
        """
        Enhanced version that first tries exact match, then fuzzy matching with suggestions.
        
        Args:
            title (str): The title of the page to find
            space_name (str, optional): The space name to search within
        
        Returns:
            dict: Contains 'page' (if found), 'suggestions' (if not found), and 'status'
        """
        # Try exact match first
        exact_match = self.get_page_by_title(title, space_name)
        
        if exact_match:
            return {
                'status': 'found',
                'page': exact_match,
                'suggestions': []
            }
        
        # If no exact match, try fuzzy matching
        suggestions = self.get_pages_by_title_fuzzy(title, space_name)
        
        if suggestions:
            return {
                'status': 'suggestions',
                'page': None,
                'suggestions': suggestions[:3]  # Limit to top 3 suggestions
            }
        
        return {
            'status': 'not_found',
            'page': None,
            'suggestions': []
        }
    
    def save_document(self, doc: Document, embedding: np.ndarray):
        """Save document with its embedding to the database."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents
                (id, title, content, space_key, space_name, type, url, last_modified, excerpt, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc.id, doc.title, doc.content, doc.space_key, doc.space_name,
                doc.type, doc.url, doc.last_modified, doc.excerpt,
                embedding.tobytes()
            ))
            # Update FTS table
            cursor.execute('''
                INSERT OR REPLACE INTO fts_documents (id, title, content, excerpt)
                VALUES (?, ?, ?, ?)
            ''', (
                doc.id, doc.title, doc.content, doc.excerpt
            ))
            self.conn.commit()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
            row = cursor.fetchone()
            if row:
                return Document(
                    id=row[0], title=row[1], content=row[2], space_key=row[3],
                    space_name=row[4], type=row[5], url=row[6], last_modified=row[7],
                    excerpt=row[8]
                )
        return None
    
    def get_document_last_modified(self, doc_id: str) -> Optional[str]:
        """Get document's last modified timestamp."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT last_modified FROM documents WHERE id = ?', (doc_id,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def get_all_document_ids(self) -> Set[str]:
        """Get all document IDs in the database."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT id FROM documents')
            return {row[0] for row in cursor.fetchall()}
    
    def get_all_documents(self, space_key: str = None) -> List[Document]:
        """Get all documents, optionally filtered by space."""
        with self.lock:
            cursor = self.conn.cursor()
            if space_key:
                cursor.execute('SELECT * FROM documents WHERE space_key = ?', (space_key,))
            else:
                cursor.execute('SELECT * FROM documents')
            
            rows = cursor.fetchall()
            return [Document(
                id=row[0], title=row[1], content=row[2], space_key=row[3],
                space_name=row[4], type=row[5], url=row[6], last_modified=row[7],
                excerpt=row[8]
            ) for row in rows]
    
    def delete_document(self, doc_id: str):
        """Delete document from database."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            cursor.execute('DELETE FROM fts_documents WHERE id = ?', (doc_id,))
            self.conn.commit()
    
    def save_space(self, space_key: str, space_name: str, description: str = ""):
        """Save space information."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO spaces (key, name, description, last_synced)
                VALUES (?, ?, ?, ?)
            ''', (space_key, space_name, description, datetime.now().isoformat()))
            self.conn.commit()
    
    def get_spaces(self) -> List[Dict]:
        """Get all spaces."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT key, name, description FROM spaces')
            return [{'key': row[0], 'name': row[1], 'description': row[2]} for row in cursor.fetchall()]
    
    def get_space_key_by_name(self, space_name: str) -> Optional[str]:
        """Get space key by space name."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT key FROM spaces WHERE LOWER(name) = LOWER(?)', (space_name,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def set_sync_metadata(self, key: str, value: str):
        """Set sync metadata."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO sync_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, datetime.now().isoformat()))
            self.conn.commit()
    
    def get_sync_metadata(self, key: str) -> Optional[str]:
        """Get sync metadata."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT value FROM sync_metadata WHERE key = ?', (key,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def semantic_search(self, query: str, space_key: str = None, limit: int = 5) -> List[Dict]:
        """Perform semantic search on documents."""
        query_embedding = embedding_model.encode([query])

        with self.lock:
            cursor = self.conn.cursor()
            if space_key:
                cursor.execute('SELECT * FROM documents WHERE space_key = ?', (space_key,))
            else:
                cursor.execute('SELECT * FROM documents')

            rows = cursor.fetchall()

            if not rows:
                return []

            embeddings = np.vstack([np.frombuffer(row[9], dtype=np.float32) for row in rows])
            similarities = cosine_similarity(query_embedding, embeddings)[0]

            results = []
            for row, sim in zip(rows, similarities):
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'space_key': row[3],
                    'space_name': row[4],
                    'type': row[5],
                    'url': row[6],
                    'last_modified': row[7],
                    'excerpt': row[8],
                    'similarity': float(sim)
                })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
    
    def add_chat_message(self, session_id: str, role: str, content: str):
        """Add message to chat history."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO chat_history (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, role, content, datetime.now().isoformat()))
            self.conn.commit()
            self.prune_chat_history(session_id)
    
    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get chat history for a session."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT role, content, timestamp FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))

            rows = cursor.fetchall()
            return [{'role': row[0], 'content': row[1], 'timestamp': row[2]} for row in reversed(rows)]

    def prune_chat_history(self, session_id: str, max_length: int = 50):
        """Keep chat history per session under a maximum length."""
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM chat_history
            WHERE id IN (
                SELECT id FROM chat_history WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT -1 OFFSET ?
            )
        ''', (session_id, max_length))
        self.conn.commit()


def sync_confluence_data():
    """Incremental sync data from Confluence to local database."""
    try:
        logger.info("Starting incremental Confluence data sync...")
        
        # Get spaces first
        spaces = confluence_api.get_spaces()
        for space in spaces:
            db_manager.save_space(space['key'], space['name'], space.get('description', ''))
        
        # Get all content from Confluence
        content_items = confluence_api.get_content_by_space()
        
        # Get existing documents from database
        existing_doc_ids = db_manager.get_all_document_ids()
        current_doc_ids = set()
        
        added_count = 0
        updated_count = 0
        skipped_count = 0
        
        docs_to_embed = []
        embed_flags = []

        for item in content_items:
            try:
                doc_id = item['id']
                current_doc_ids.add(doc_id)
                
                # Get last modified timestamp from Confluence
                confluence_last_modified = item['history']['lastUpdated']['when']
                
                # Check if document exists and if it's been modified
                existing_last_modified = db_manager.get_document_last_modified(doc_id)
                
                # Skip if document hasn't changed
                if existing_last_modified and existing_last_modified == confluence_last_modified:
                    skipped_count += 1
                    continue
                
                # Extract content
                content = ""
                if item.get('body') and item['body'].get('storage'):
                    content = item['body']['storage']['value']
                
                # Create document
                doc = Document(
                    id=item['id'],
                    title=item['title'],
                    content=content,
                    space_key=item['space']['key'],
                    space_name=item['space']['name'],
                    type=item['type'],
                    url=f"{CONFLUENCE_URL}/spaces/{item['space']['key']}/pages/{item['id']}",
                    last_modified=confluence_last_modified,
                    excerpt=content[:300] if content else item['title']
                )
                
                logger.info(f"Queueing document for embedding: {doc.title} ({doc_id})")
                docs_to_embed.append(doc)
                embed_flags.append(bool(existing_last_modified))
                
            except Exception as e:
                logger.error(f"Error syncing document {item['id']}: {e}")
                continue
        
        # Batch embedding computation
        if docs_to_embed:
            texts = [f"{d.title} {d.content}" for d in docs_to_embed]
            embeddings = embedding_model.encode(texts, batch_size=32)
            for doc, emb, existed in zip(docs_to_embed, embeddings, embed_flags):
                db_manager.save_document(doc, emb)
                if existed:
                    updated_count += 1
                    logger.info(f"Updated document: {doc.title}")
                else:
                    added_count += 1
                    logger.info(f"Added new document: {doc.title}")

        # Handle deleted documents (exist in DB but not in Confluence)
        deleted_doc_ids = existing_doc_ids - current_doc_ids
        deleted_count = 0
        
        for doc_id in deleted_doc_ids:
            try:
                doc = db_manager.get_document(doc_id)
                if doc:
                    logger.info(f"Deleting document: {doc.title} ({doc_id})")
                    db_manager.delete_document(doc_id)
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting document {doc_id}: {e}")
        
        # Update sync metadata
        db_manager.set_sync_metadata('last_sync_timestamp', datetime.now().isoformat())
        
        sync_stats = {
            "added": added_count,
            "updated": updated_count,
            "deleted": deleted_count,
            "skipped": skipped_count,
            "total_processed": len(content_items)
        }
        
        logger.info(f"Incremental sync completed. Added: {added_count}, Updated: {updated_count}, "
                   f"Deleted: {deleted_count}, Skipped: {skipped_count}")
        
        return sync_stats
        
    except Exception as e:
        logger.error(f"Error during incremental sync: {e}")
        return {"error": str(e)}


def force_full_sync():
    """Force a full sync (useful for initial setup or when things go wrong)."""
    try:
        logger.info("Starting full Confluence data sync...")
        
        # Clear sync metadata to force full sync
        db_manager.set_sync_metadata('force_full_sync', 'true')
        
        # Get all existing documents and delete them
        existing_docs = db_manager.get_all_documents()
        for doc in existing_docs:
            db_manager.delete_document(doc.id)
        
        logger.info(f"Cleared {len(existing_docs)} existing documents")
        
        # Now run normal sync (will process all documents as new)
        result = sync_confluence_data()
        
        # Clear force sync flag
        db_manager.set_sync_metadata('force_full_sync', 'false')
        
        return result
        
    except Exception as e:
        logger.error(f"Error during full sync: {e}")
        return {"error": str(e)}


def get_sync_status():
    """Get current sync status and statistics."""
    try:
        last_sync = db_manager.get_sync_metadata('last_sync_timestamp')
        total_docs = len(db_manager.get_all_documents())
        spaces = db_manager.get_spaces()
        
        return {
            "last_sync": last_sync,
            "total_documents": total_docs,
            "total_spaces": len(spaces),
            "spaces": spaces
        }
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        return {"error": str(e)}


def start_background_sync():
    """Start background sync scheduler."""
    schedule.every(10).minutes.do(sync_confluence_data)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    logger.info("Background sync scheduler started")

class ConfluenceAPI:
    def __init__(self):
        self.base_url = CONFLUENCE_URL
        self.auth = HTTPBasicAuth(CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500,502,503,504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def get_spaces(self) -> List[Dict]:
        """Get all spaces from Confluence."""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/space",
                auth=self.auth,
                headers=self.headers,
                params={'limit': 100},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get('results', [])
        except requests.RequestException as e:
            status = e.response.status_code if e.response else 'N/A'
            logger.error(f"Error getting spaces ({status}): {e}")
            return []
    
    def get_content_by_space(self, space_key: str = None) -> List[Dict]:
        """Get content from Confluence, optionally filtered by space."""
        try:
            params = {'limit': 100}
            if space_key:
                params['spaceKey'] = space_key

            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                auth=self.auth,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            items = response.json().get('results', [])
            page_ids = [item['id'] for item in items]

            pages = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                for page in executor.map(self.get_page_by_id, page_ids):
                    if page:
                        pages.append(page)

            return pages
        except requests.RequestException as e:
            status = e.response.status_code if e.response else 'N/A'
            logger.error(f"Error getting content ({status}): {e}")
            return []
    
    def get_page_by_id(self, page_id: str) -> Optional[Dict]:
        """Get page details by ID."""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{page_id}",
                auth=self.auth,
                headers=self.headers,
                params={'expand': 'body.storage,version,space,history.lastUpdated'},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            status = e.response.status_code if e.response else 'N/A'
            logger.error(f"Error getting page {page_id} ({status}): {e}")
            return None
    
    def create_page(self, space_key: str, title: str, content: str, parent_id: str = None) -> Dict:
        """Create a new page in Confluence."""
        try:
            data = {
                'type': 'page',
                'title': title,
                'space': {'key': space_key},
                'body': {
                    'storage': {
                        'value': content,
                        'representation': 'storage'
                    }
                }
            }
            
            if parent_id:
                data['ancestors'] = [{'id': parent_id}]
            
            response = self.session.post(
                f"{self.base_url}/rest/api/content",
                auth=self.auth,
                headers=self.headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            status = e.response.status_code if e.response else 'N/A'
            logger.error(f"Error creating page ({status}): {e}")
            raise
    
    def update_page(self, page_id: str, title: str, content: str, version: int) -> Dict:
        """Update an existing page in Confluence."""
        try:
            data = {
                'version': {'number': version + 1},
                'title': title,
                'type': 'page',
                'body': {
                    'storage': {
                        'value': content,
                        'representation': 'storage'
                    }
                }
            }
            
            response = self.session.put(
                f"{self.base_url}/rest/api/content/{page_id}",
                auth=self.auth,
                headers=self.headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            status = e.response.status_code if e.response else 'N/A'
            logger.error(f"Error updating page ({status}): {e}")
            raise
    
    def delete_page(self, page_id: str) -> bool:
        """Delete a page from Confluence."""
        try:
            response = self.session.delete(
                f"{self.base_url}/rest/api/content/{page_id}",
                auth=self.auth,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            status = e.response.status_code if e.response else 'N/A'
            logger.error(f"Error deleting page ({status}): {e}")
            return False

# Initialize components
db_manager = DatabaseManager()
confluence_api = ConfluenceAPI()

class AIAssistant:
    def __init__(self):
        self.functions = [
            {
                "name": "search_confluence",
                "description": "Search for relevant documents in Confluence based on user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "space_key": {
                            "type": "string",
                            "description": "Optional space key to limit search to specific space"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "create_confluence_page",
                "description": "Create a new page in Confluence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "space_name": {
                            "type": "string",
                            "description": "The space name where to create the page (user-friendly name, not key)"
                        },
                        "page_title": {
                            "type": "string",
                            "description": "The title of the new page"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content of the new page in Confluence storage format"
                        },
                        "parent_id": {
                            "type": "string",
                            "description": "Optional parent page ID"
                        }
                    },
                    "required": ["space_name", "page_title", "content"]
                }
            },
            {
                "name": "update_confluence_page",
                "description": "Update an existing page in Confluence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "string",
                            "description": "The title of the page to update"
                        },
                        "space_name": {
                            "type": "string",
                            "description": "The space name where the page is located (optional, helps with disambiguation)"
                        },
                        "new_title": {
                            "type": "string",
                            "description": "The new title of the page (optional, if changing the title)"
                        },
                        "content": {
                            "type": "string",
                            "description": "The new content of the page"
                        }
                    },
                    "required": ["page_title", "content"]
                }
            },
            {
                "name": "delete_confluence_page",
                "description": "Delete a page from Confluence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "string",
                            "description": "The title of the page to delete"
                        },
                        "space_name": {
                            "type": "string",
                            "description": "The space name where the page is located (optional, helps with disambiguation)"
                        }
                    },
                    "required": ["page_title"]
                }
            },
        ]
    
    def get_response(self, message: str, session_id: str, space_key: str = None) -> Dict:
        """Get AI response with function calling."""
        try:
            # Get chat history for context
            chat_history = db_manager.get_chat_history(session_id)
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": sysprompt
                }
            ]
            
            # Add chat history
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[{"type": "function", "function": func} for func in self.functions],
                temperature=0.7
            )
            
            message_response = response.choices[0].message

            # Handle tool calls
            if message_response.tool_calls:
                return self._handle_tool_call(message_response.tool_calls[0], session_id, space_key)
            else:
                return {
                    "response": message_response.content,
                    "search_performed": False
                }
        
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "search_performed": False
            }
    
    def _handle_tool_call(self, tool_call, session_id: str, space_key: str = None) -> Dict:
        """Handle function calls from OpenAI."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "search_confluence":
            query = function_args["query"]
            search_space = function_args.get("space_key", space_key)
            
            results = db_manager.semantic_search(query, search_space, limit=3)
            
            if results:
                # Generate AI response based on search results
                context = "\n\n".join([f"Title: {result['title']}\nContent: {result['content'][:5000]}" for result in results])
                prompt = f"""
Based ONLY on the provided Confluence documents, answer the user's question. 

STRICT RULES:
1. ONLY use information that exists in the provided documents
2. If the documents don't contain the answer, say "The documents don't contain information about this topic. The relevant documents will be shown below."
3. DO NOT make up, invent, or add any information not in the documents
4. Start your response by mentioning which page(s) contain the relevant information
5. If you find relevant information, end with "Check out the relevant documents below!"
6. Answer thoroughly the user questions

User question: {query}

Context from Confluence documents:
{context}

Answer based ONLY on the documents above:
                """
                
                # Create a new chat completion to answer the question with context
                answer_messages = [
                    {
                        "role": "system",
                        "content": prompt
                    },
                ]
                
                answer_response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=answer_messages,
                    temperature=0.4  # Lower temperature for more consistent responses
                )
                
                ai_answer = answer_response.choices[0].message.content
                
                return {
                    "response": ai_answer,
                    "search_performed": True,
                    "search_query": query,
                    "results": results,
                    "total_results": len(results)
                }
            else:
                return {
                    "response": f"I couldn't find any documents related to '{query}'. Try rephrasing your search or check if the content exists in Confluence.",
                    "search_performed": True,
                    "search_query": query,
                    "results": [],
                    "total_results": 0
                }
        
        elif function_name == "create_confluence_page":
            # Convert space name to space key
            space_name = function_args.get('space_name')
            if space_name:
                space_key = db_manager.get_space_key_by_name(space_name)
                if not space_key:
                    available_spaces = db_manager.get_spaces()
                    space_list = "\n".join([f"- {space['name']}" for space in available_spaces])
                    return {
                        "response": f"I couldn't find a space named '{space_name}'. Available spaces are:\n\n{space_list}\n\nPlease specify one of these space names.",
                        "search_performed": False
                    }
                function_args['space_key'] = space_key
            
            return {
                "response": f"I want to create a new page with the following details:\n\n"
                          f"**Title**: {function_args['page_title']}\n"
                          f"**Space**: {space_name}\n"
                          f"**Content**: {function_args['content'][:100]}...\n\n"
                          f"Do you want me to proceed? Please reply 'yes' to confirm or 'no' to cancel.",
                "search_performed": False,
                "pending_action": {
                    "type": "create_page",
                    "args": function_args
                }
            }
        
        elif function_name == "update_confluence_page":
            page_title = function_args['page_title']
            space_name = function_args.get('space_name', '')
            new_title = function_args.get('new_title', page_title)
            
            # Get page info with suggestions
            page_info = db_manager.get_page_by_title_with_suggestions(page_title, space_name)
            
            # Check if page was found
            if page_info['status'] == 'not_found':
                return {
                    "response": f"I couldn't find a page titled '{page_title}'" + 
                            (f" in space '{space_name}'" if space_name else "") + 
                            ". Please check the title and try again.",
                    "search_performed": False
                }
            
            # If we have suggestions but no exact match
            if page_info['status'] == 'suggestions':
                suggestions_text = "\n".join([f"- {s['title']} (in {s['space_name']})" 
                                            for s in page_info['suggestions']])
                return {
                    "response": f"I couldn't find an exact match for '{page_title}'" + 
                            (f" in space '{space_name}'" if space_name else "") + 
                            f". Did you mean one of these pages?\n\n{suggestions_text}",
                    "search_performed": False
                }
            
            # Page found - extract the actual page data
            page_data = page_info['page']
            
            # Add the page_id to the args for the actual update
            function_args['page_id'] = page_data['id']
            
            return {
                "response": f"I want to update the page '{page_title}'" + 
                        (f" in space '{space_name}'" if space_name else "") + ":\n\n"
                        f"**New Title**: {new_title}\n"
                        f"**New Content**: {function_args['content'][:100]}...\n\n"
                        f"Do you want me to proceed? Please reply 'yes' to confirm or 'no' to cancel.",
                "search_performed": False,
                "pending_action": {
                    "type": "update_page",
                    "args": function_args
                }
            }
        
        elif function_name == "delete_confluence_page":
            page_title = function_args['page_title']
            space_name = function_args.get('space_name', '')
            
            # You'll need to implement a method to find page by title
            page_info = db_manager.get_page_by_title_with_suggestions(page_title, space_name)
            
            if not page_info:
                return {
                    "response": f"I couldn't find a page titled '{page_title}'" + 
                              (f" in space '{space_name}'" if space_name else "") + 
                              ". Please check the title and try again.",
                    "search_performed": False
                }
            
            # Page found - extract the actual page data
            page_data = page_info['page']
            
            # Add the page_id to the args for the actual update
            function_args['page_id'] = page_data['id']
            
            return {
                "response": f"I want to delete the page '{page_title}'" + 
                          (f" in space '{space_name}'" if space_name else "") + ".\n\n"
                          f"⚠️ **Warning**: This action cannot be undone!\n\n"
                          f"Do you want me to proceed? Please reply 'yes' to confirm or 'no' to cancel.",
                "search_performed": False,
                "pending_action": {
                    "type": "delete_page",
                    "args": function_args
                }
            }
        
        return {
            "response": "I'm not sure how to handle that request.",
            "search_performed": False
        }   
             
# Initialize AI assistant
ai_assistant = AIAssistant()

# Routes
@app.route('/')
def index():
    """Main chat interface."""
    if 'session_id' not in session:
        session['session_id'] = str(int(time.time() * 1000))
    return render_template('index.html')

@app.route('/api/spaces')
def get_spaces():
    """Get all available spaces."""
    try:
        spaces = db_manager.get_spaces()
        return jsonify({"spaces": spaces})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.json
        message = data.get('message', '').strip()
        space_key = data.get('space_key')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(int(time.time() * 1000))
            session_id = session['session_id']
        
        # Save user message to history
        db_manager.add_chat_message(session_id, 'user', message)
        
        # Check if this is a confirmation for a pending action
        if message.lower() in ['yes', 'y', 'ya', 'iya']:
            # Handle pending action confirmation
            pending_action = session.get('pending_action')
            if pending_action:
                result = execute_pending_action(pending_action)
                session.pop('pending_action', None)
                db_manager.add_chat_message(session_id, 'assistant', result['response'])
                return jsonify(result)
        
        elif message.lower() in ['no', 'n', 'tidak', 'cancel']:
            # Cancel pending action
            session.pop('pending_action', None)
            response = "Action cancelled."
            db_manager.add_chat_message(session_id, 'assistant', response)
            return jsonify({"response": response, "search_performed": False})
        
        # Get AI response
        response = ai_assistant.get_response(message, session_id, space_key)
        
        # Save pending action to session if any
        if 'pending_action' in response:
            session['pending_action'] = response['pending_action']
        
        # Save assistant response to history
        db_manager.add_chat_message(session_id, 'assistant', response['response'])
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def execute_pending_action(pending_action):
    """Execute a pending CRUD action."""
    try:
        action_type = pending_action['type']
        args = pending_action['args']
        
        if action_type == 'create_page':
            result = confluence_api.create_page(
                args['space_key'], 
                args['page_title'], 
                args['content'],
                args.get('parent_id')
            )
            return {
                "response": f"✅ Page '{args['page_title']}' created successfully! You can view it [right here!]({CONFLUENCE_URL}{result['_links']['webui']})",
                "search_performed": False
            }
        
        elif action_type == 'update_page':
            # Get current page info from Confluence API
            page_info = confluence_api.get_page_by_id(args['page_id'])
            if not page_info:
                return {
                    "response": "❌ Page not found. Cannot update. Please check if the page ID is correct.",
                    "search_performed": False
                }
            
            # Update page with correct version number
            current_version = page_info['version']['number']
            result = confluence_api.update_page(
                args['page_id'],
                args['page_title'],
                args['content'],
                current_version
            )
            return {
                "response": f"✅ Page '{args['page_title']}' updated successfully! You can view it at: {page_info['_links']['webui']}",
                "search_performed": False
            }
        
        elif action_type == 'delete_page':
            # First check if page exists
            page_info = confluence_api.get_page_by_id(args['page_id'])
            if not page_info:
                return {
                    "response": "❌ Page not found. Cannot delete. Please check if the page ID is correct.",
                    "search_performed": False
                }
            
            success = confluence_api.delete_page(args['page_id'])
            if success:
                db_manager.delete_document(args['page_id'])
                return {
                    "response": f"✅ Page '{args['page_title']}' deleted successfully!",
                    "search_performed": False
                }
            else:
                return {
                    "response": "❌ Failed to delete page. Please check if you have permissions.",
                    "search_performed": False
                }
        
        return {
            "response": "❌ Unknown action type.",
            "search_performed": False
        }
        
    except Exception as e:
        logger.error(f"Error executing pending action: {e}")
        return {
            "response": f"❌ Error executing action: {str(e)}",
            "search_performed": False
        }

@app.route('/api/sync/incremental', methods=['POST'])
def sync_incremental():
    """Trigger incremental sync."""
    try:
        stats = sync_confluence_data()
        return jsonify({"message": "Sync completed successfully", "stats": stats})
    except Exception as e:
        logger.error(f"Error during manual sync: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start background sync
    start_background_sync()
    
    # Initial sync
    logger.info("Performing initial sync...")
    sync_confluence_data()
    
    app.run(host='0.0.0.0', port=5000)
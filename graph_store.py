import json
import sqlite3
import os
import argparse
import networkx as nx
from typing import Dict, Any, List
from abc import ABC, abstractmethod

# ---------------------------------------------------------
# Pluggable Storage Driver Protocol
# ---------------------------------------------------------

class StorageProvider(ABC):
    @abstractmethod
    def init_db(self, db_path: str):
        pass
        
    @abstractmethod
    def save_entity(self, entity: Dict[str, Any]):
        pass
        
    @abstractmethod
    def save_claim(self, claim: Dict[str, Any]) -> str:
        pass
        
    @abstractmethod
    def save_evidence(self, claim_id: str, evidence: Dict[str, Any]):
        pass
        
    @abstractmethod
    def commit(self):
        pass

    @abstractmethod
    def log_search(self, query: str, hits: List[str]):
        pass

# ---------------------------------------------------------
# SQLite Implementation
# ---------------------------------------------------------

class SQLiteStorageProvider(StorageProvider):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = self.init_db(db_path)
        
    def init_db(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                aliases TEXT,
                valid_from TEXT,
                valid_to TEXT,
                version INTEGER DEFAULT 1,
                PRIMARY KEY (id, version)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT,
                subject_id TEXT,
                predicate TEXT,
                object_id TEXT,
                valid_at TEXT,
                invalid_at TEXT,
                expired_at TEXT,
                valid_from TEXT,
                valid_to TEXT,
                version INTEGER DEFAULT 1,
                confidence_score REAL DEFAULT 1.0,
                last_observed_at TEXT,
                PRIMARY KEY (id, version),
                FOREIGN KEY(subject_id) REFERENCES entities(id),
                FOREIGN KEY(object_id) REFERENCES entities(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evidences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT,
                source_id TEXT,
                exact_excerpt TEXT,
                character_start_offset INTEGER,
                character_end_offset INTEGER,
                timestamp TEXT,
                FOREIGN KEY(claim_id) REFERENCES claims(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                hits_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS community_reports (
                id TEXT PRIMARY KEY,
                title TEXT,
                summary TEXT,
                node_ids TEXT
            )
        ''')
        
        conn.commit()
        return conn

    def save_entity(self, entity: Dict[str, Any]):
        import datetime
        now_ts = datetime.datetime.now().isoformat()
        cursor = self.conn.cursor()
        aliases_str = json.dumps(entity.get("aliases", []))
        
        cursor.execute("SELECT version FROM entities WHERE id = ? AND valid_to IS NULL", (entity["id"],))
        result = cursor.fetchone()
        
        if result:
            version = result[0] + 1
            cursor.execute("UPDATE entities SET valid_to = ? WHERE id = ? AND valid_to IS NULL", (now_ts, entity["id"]))
            cursor.execute('''
                INSERT INTO entities (id, name, type, aliases, valid_from, version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (entity["id"], entity["name"], entity["type"], aliases_str, now_ts, version))
        else:
            cursor.execute('''
                INSERT INTO entities (id, name, type, aliases, valid_from, version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (entity["id"], entity["name"], entity["type"], aliases_str, now_ts, 1))
        
    def save_claim(self, claim: Dict[str, Any]) -> str:
        import datetime
        now_ts = datetime.datetime.now().isoformat()
        cursor = self.conn.cursor()
        valid_at = str(claim.get("valid_at", claim.get("temporal_validity", "Always")))
        invalid_at = str(claim.get("invalid_at", ""))
        expired_at = str(claim.get("expired_at", ""))
        
        claim_id = claim.get("claim_id", f"claim_{id(claim)}")
        
        # Memento confidence properties
        confidence_score = float(claim.get("confidence_score", 1.0))
        last_observed_at = str(claim.get("last_observed_at", now_ts))
        
        cursor.execute("SELECT version FROM claims WHERE id = ? AND valid_to IS NULL", (claim_id,))
        result = cursor.fetchone()
        
        if result:
            version = result[0] + 1
            cursor.execute("UPDATE claims SET valid_to = ? WHERE id = ? AND valid_to IS NULL", (now_ts, claim_id))
            cursor.execute('''
                INSERT INTO claims (id, subject_id, predicate, object_id, valid_at, invalid_at, expired_at, valid_from, version, confidence_score, last_observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (claim_id, claim["subject_id"], claim["predicate"], claim["object_id"], valid_at, invalid_at, expired_at, now_ts, version, confidence_score, last_observed_at))
        else:
            cursor.execute('''
                INSERT INTO claims (id, subject_id, predicate, object_id, valid_at, invalid_at, expired_at, valid_from, version, confidence_score, last_observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (claim_id, claim["subject_id"], claim["predicate"], claim["object_id"], valid_at, invalid_at, expired_at, now_ts, 1, confidence_score, last_observed_at))
            
        return claim_id

    def save_evidence(self, claim_id: str, evidence: Dict[str, Any]):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO evidences (claim_id, source_id, exact_excerpt, character_start_offset, character_end_offset, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            claim_id, 
            evidence["source_id"], 
            evidence["exact_excerpt"], 
            evidence["character_start_offset"], 
            evidence["character_end_offset"], 
            evidence.get("timestamp", "")
        ))

    def commit(self):
        self.conn.commit()

    def log_search(self, query: str, hits: List[str]):
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO search_logs (timestamp, query, hits_json)
            VALUES (?, ?, ?)
        ''', (timestamp, query, json.dumps(hits)))
        self.conn.commit()

# ---------------------------------------------------------
# Graph Construction & Persistence
# ---------------------------------------------------------

class GraphStore:
    def __init__(self, driver: StorageProvider):
        self.driver = driver
        self.graph = nx.MultiDiGraph()
        
    def ingest_canonical_data(self, data: Dict[str, Any]):
        """Persists the canonicalized JSON graph into the driver and initializes the NetworkX graph."""
        print("Persisting entities...")
        for entity in data.get("entities", []):
            self.driver.save_entity(entity)
            self.graph.add_node(entity["id"], name=entity["name"], type=entity["type"], aliases=entity.get("aliases", []))
            
        print("Persisting claims and evidence...")
        for claim in data.get("claims", []):
            claim_id = self.driver.save_claim(claim)
            
            evidences = claim.get("evidences", [])
            for ev in evidences:
                self.driver.save_evidence(claim_id, ev)
            
            valid_at = str(claim.get("valid_at", claim.get("temporal_validity", "Always")))
            invalid_at = str(claim.get("invalid_at", ""))
            
            self.graph.add_edge(
                claim["subject_id"], 
                claim["object_id"], 
                key=claim_id,
                predicate=claim["predicate"], 
                valid_at=valid_at,
                invalid_at=invalid_at,
                evidence_count=len(evidences)
            )
            
        self.driver.commit()

# ---------------------------------------------------------
# Serialization
# ---------------------------------------------------------

def export_graph(graph: nx.MultiDiGraph, export_path: str):
    """Serializes the NetworkX graph state to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    data = nx.node_link_data(graph)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Build and Persist Memory Graph")
    parser.add_argument("--input", type=str, default="data/canonical.json", help="Canonicalized JSON input")
    parser.add_argument("--db", type=str, default="data/graph.db", help="SQLite database path")
    parser.add_argument("--export", type=str, default="data/graph_export.json", help="Final NetworkX JSON export path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    print(f"Loading canonical data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
        
    store = GraphStore(SQLiteStorageProvider(args.db))
    store.ingest_canonical_data(data)
    
    print(f"Graph loaded into SQLite at {args.db}")
    print(f"In-memory Graph: {store.graph.number_of_nodes()} nodes, {store.graph.number_of_edges()} edges.")
    
    print(f"Exporting structure to {args.export}...")
    export_graph(store.graph, args.export)
    print("Done.")

if __name__ == "__main__":
    main()

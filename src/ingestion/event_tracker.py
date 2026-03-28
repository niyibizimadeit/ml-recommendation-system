import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

# Path to the database file
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "logs" / "interactions.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables if they don't exist yet."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT    NOT NULL,
                product_id  TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,  -- view, add_to_cart, purchase, search
                weight      REAL    NOT NULL,  -- signal strength (see WEIGHTS)
                metadata    TEXT,              -- optional JSON blob (search query, etc.)
                timestamp   TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id  TEXT PRIMARY KEY,
                name        TEXT,
                category    TEXT,
                price       REAL,
                tags        TEXT,   -- comma-separated
                description TEXT,
                created_at  TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user 
            ON interactions(user_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_product 
            ON interactions(product_id)
        """)
        print(f"Database initialised at {DB_PATH}")


# Signal weights — tune these as you learn more about your users
WEIGHTS = {
    "view":        1.0,
    "add_to_cart": 3.0,
    "purchase":    5.0,
    "search":      1.5,
}


def log_event(user_id: str, product_id: str, event_type: str, metadata: dict = None):
    """
    Log a single user interaction.

    Args:
        user_id:    Unique identifier for the user (session ID is fine early on)
        product_id: Your internal product ID
        event_type: One of 'view', 'add_to_cart', 'purchase', 'search'
        metadata:   Optional dict — e.g. {"query": "blue sneakers", "time_on_page": 45}
    """
    if event_type not in WEIGHTS:
        raise ValueError(f"Unknown event_type '{event_type}'. Use: {list(WEIGHTS.keys())}")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO interactions (user_id, product_id, event_type, weight, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                product_id,
                event_type,
                WEIGHTS[event_type],
                json.dumps(metadata) if metadata else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def log_product(product_id: str, name: str, category: str, price: float,
                tags: list[str] = None, description: str = ""):
    """Add or update a product in the catalogue."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO products (product_id, name, category, price, tags, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                name=excluded.name,
                category=excluded.category,
                price=excluded.price,
                tags=excluded.tags,
                description=excluded.description
            """,
            (
                product_id,
                name,
                category,
                price,
                ",".join(tags) if tags else "",
                description,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_interactions(limit: int = 100):
    """Quick look at recent interactions — useful for debugging."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    # Run this file directly to initialise the DB and insert test data
    init_db()

    # Seed a few fake interactions to confirm everything works
    log_product("P001", "White Sneakers", "Footwear", 89.99,
                tags=["shoes", "casual", "white"], description="Classic white sneakers")
    log_product("P002", "Black Hoodie",   "Clothing",  49.99,
                tags=["hoodie", "casual", "black"], description="Comfortable black hoodie")
    log_product("P003", "Running Shorts", "Clothing",  34.99,
                tags=["shorts", "sport", "running"], description="Lightweight running shorts")

    log_event("user_001", "P001", "view")
    log_event("user_001", "P001", "add_to_cart")
    log_event("user_001", "P002", "view")
    log_event("user_002", "P001", "view",        metadata={"time_on_page": 62})
    log_event("user_002", "P003", "purchase",    metadata={"order_id": "ORD-001"})
    log_event("user_003", "P002", "add_to_cart")
    log_event("user_003", "P003", "view")

    print("\nRecent interactions:")
    for row in get_interactions():
        print(f"  {row['timestamp'][:19]}  {row['user_id']}  {row['product_id']}  {row['event_type']}  (weight={row['weight']})")
import uuid
import numpy as np
from app.utils.cache import FaceEmbeddingCache

def test_vector_cache_matching() -> None:
    """Verifies that local VectorCache properly inserts and queries embeddings."""
    cache = FaceEmbeddingCache()
    cache.clear()
    
    # 1. Generate 2 synthetic 512-dimension unit vector embeddings
    emb_alice = np.zeros(512, dtype=np.float32)
    emb_alice[0] = 1.0  # Alice's face representation direction
    
    emb_bob = np.zeros(512, dtype=np.float32)
    emb_bob[1] = 1.0    # Bob's face representation direction
    
    alice_uuid = uuid.uuid4()
    bob_uuid = uuid.uuid4()
    
    # Register in cache
    cache.add(alice_uuid, "Alice Smith", "STD001", emb_alice)
    cache.add(bob_uuid, "Bob Jones", "STD002", emb_bob)
    
    assert cache.size == 2

    # 2. Search for Alice using exact match vector
    match = cache.search(emb_alice, threshold=0.5)
    assert match is not None
    assert match[0] == alice_uuid
    assert match[1] == "Alice Smith"
    assert match[2] == "STD001"
    assert match[3] >= 0.99  # Score should be ~1.0

    # 3. Search with Bob's vector
    match_bob = cache.search(emb_bob, threshold=0.5)
    assert match_bob is not None
    assert match_bob[0] == bob_uuid
    assert match_bob[1] == "Bob Jones"

    # 4. Search with completely orthogonal face vector (should be similarity ~0.0, below threshold)
    emb_unknown = np.zeros(512, dtype=np.float32)
    emb_unknown[2] = 1.0
    
    match_unknown = cache.search(emb_unknown, threshold=0.50)
    assert match_unknown is None  # Should return Unknown/None

    # 5. Remove a student from cache
    cache.remove(alice_uuid)
    assert cache.size == 1
    
    match_removed = cache.search(emb_alice, threshold=0.50)
    assert match_removed is None

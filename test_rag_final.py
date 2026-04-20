"""
Final test of RAG engine with real PostgreSQL + pgvector (no mock)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend_api.app.engine import RAGEngine
from backend_api.app.config import Settings
from data_ingestion.vector_store import VectorStore
from data_ingestion.embedder import EmbeddingGenerator

# Create settings
settings = Settings(
    database_url="postgresql://atlas:atlas@localhost:5432/atlas_rag",
    vector_table="atlas_knowledge",
    embedding_model="BAAI/bge-m3",
    embedding_dimension=1024,
    similarity_threshold=0.4,  # Lower threshold
    top_k_results=5,
    temperature=0.0
)

# Create engine with mock LLM but real PostgreSQL vector store and embeddings
engine = RAGEngine(settings=settings, use_mock=True)

# Override vector store and embedder to use real ones (not mock)
engine._vector_store = VectorStore(
    dsn=settings.database_url,
    table_name=settings.vector_table,
    embedding_dimension=settings.embedding_dimension
)

engine._embedder = EmbeddingGenerator(model_name=settings.embedding_model)

def test_query(question: str):
    """Test a query directly."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    try:
        response = engine.query(question)
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence:.2f}")
        print(f"\nCitations ({len(response.citations)}):")
        for i, citation in enumerate(response.citations, 1):
            print(f"  {i}. {citation.title}")
            print(f"     URL: {citation.url}")
            print(f"     Score: {citation.score:.2f}")
            print(f"     Snippet: {citation.text_snippet[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing Atlas-RAG Engine with real PostgreSQL + pgvector and embeddings\n")
    
    # Test queries about the law
    test_query("Quelles sont les conditions pour obtenir une autorisation d'exercice d'assurance?")
    test_query("Quelles sont les sanctions en cas de violation de la loi sur les assurances?")
    
    # Test queries about the portal guide
    test_query("Comment récupérer mon mot de passe oublié sur le portail?")
    test_query("Comment soumettre un formulaire sur le portail ACAPS?")
    
    print("\n" + "="*60)
    print("Testing complete!")


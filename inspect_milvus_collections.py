#!/usr/bin/env python3
"""Quick script to inspect Milvus collections"""

from pymilvus import MilvusClient

db_path = "experiment_results/milvus_retail.db"
client = MilvusClient(uri=db_path)

print(f"Available collections: {client.list_collections()}")

for collection_name in client.list_collections():
    print(f"\nCollection: {collection_name}")
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
        print(f"  Stats: {stats}")
        
        # Sample one entity
        result = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["*"],
            limit=1
        )
        if result:
            print(f"  Sample entity keys: {list(result[0].keys())}")
            print(f"  Sample entity: {result[0]}")
    except Exception as e:
        print(f"  Error: {e}")

client.close()

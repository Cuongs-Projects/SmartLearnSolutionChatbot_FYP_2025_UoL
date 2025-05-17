#!/usr/bin/env python3
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

# --- 1. Configuration ---
CHROMA_DB_PATH = "PADBRC/PADBRC_vector_DB"
COLLECTION_NAME = "smartlearn_padbrc"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

base_dir = "PADBRC"
transcribed_contents = "transcribed_contents"

class SCCDB_SML():
    # --- 2. Initialize ChromaDB Client and Embedding Model ---
    @staticmethod
    def initialise_chromadb_and_embedding_model(chromadb_path,embed_model_name):
        try:
            client = chromadb.PersistentClient(path=chromadb_path)
            print(f"ChromaDB client initialized. Data will be stored in: {chromadb_path}")
        except Exception as e:
            print(f"Error initializing ChromaDB client: {e}")
            exit()

        try:
            embedding_model = SentenceTransformer(embed_model_name)
            print(f"Embedding model '{embed_model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            exit()
        
        return client,embedding_model
    
    # --- 3. Create or Get Collection ---
    @staticmethod
    def get_create_collection(client,collection_name):
        collection = client.get_or_create_collection(name=collection_name)
        print(f"Collection '{collection_name}' ready.")
        return collection
    
    # --- 3.1. Delete Collection ---
    #COLLECTION_NAME_TO_DELETE = "smartlearn_padbrc" # The specific collection you want to reset

    @staticmethod
    def delete_the_collection(chromadb_path,collection_name):
        client = chromadb.PersistentClient(path=chromadb_path)

        try:
            print(f"Attempting to delete collection: {collection_name}")
            client.delete_collection(name=collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            # Chroma might raise an error if the collection doesn't exist,
            # or you might get a different error for other issues.
            # Depending on the exact error, you might choose to ignore "collection not found"
            # or handle it more specifically. For a simple reset, often it's fine if it wasn't there.
            print(f"Note or Error deleting collection '{collection_name}': {e}")
            print("This might be okay if the collection didn't exist in the first place.")

    #delete_the_collection(chromadb_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME_TO_DELETE)

    # --- 4. Load Your Prepared JSON Chunk Data ---
    @staticmethod
    def load_json_chunks(base_dir, transcribed_contents):
        files_path = os.path.join(base_dir,transcribed_contents)
        all_chunks_data = []
        entries = os.listdir(files_path)
        for file in entries:
            if file.endswith(".json"):
                file_path = os.path.join(files_path,file)
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                for row in data:
                    all_chunks_data.append(row)
        if not all_chunks_data:
            print("No data loaded. Exiting.")
            exit()
        return all_chunks_data
    
    # --- 5. Prepare Data for ChromaDB Batch Addition ---
    @staticmethod
    def db_batch_prep(all_chunks_data):
        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        section_dict = {"S0":"Preparation","S1":"Starting_Exercise","S2":"Module_1"}

        print(f"Processing {len(all_chunks_data)} chunks for embedding...")
        for chunk_data in all_chunks_data:
            text_to_embed = chunk_data.get("text", "")
            if not text_to_embed.strip():
                continue
            documents_to_add.append(text_to_embed)
            
            ind = chunk_data.get("id")
            parts = ind.split('_')
            current_metadata = {
                "source_type": chunk_data.get("source_type"),
                "source_identifier": chunk_data.get("source_identifier"),
                "visual_description": chunk_data.get("visual_description", None),
                "title" : parts[-2],
                "section" : section_dict[parts[1]],
                "subsection" : "Exercise_"+parts[2][2:],
                "step":"Activity_"+parts[3][2:]
            }

            if chunk_data.get("source_type") == "video":
                current_metadata["start_sec"] = chunk_data.get("start_sec")
                current_metadata["end_sec"] = chunk_data.get("end_sec")
            elif chunk_data.get("source_type") == "pdf":
                current_metadata["page_number"] = chunk_data.get("page_number")

            cleaned_metadata = {}
            for key, value in current_metadata.items():
                if value is None:
                    cleaned_metadata[key] = "N/A"
                elif isinstance(value,(str,int,bool,float)):
                    cleaned_metadata[key] = value

            metadatas_to_add.append(cleaned_metadata)
            ids_to_add.append(str(chunk_data.get("id")))
            
        return documents_to_add, metadatas_to_add, ids_to_add
    
    # --- 6. Generate Embeddings (in batches if dataset is large) ---
    @staticmethod
    def generate_embedding(documents_to_add, metadatas_to_add, ids_to_add,embedding_model,collection,collection_name): #collection --> the variable, collection_name --> str
        print(f"Generating embeddings for {len(documents_to_add)} documents...")
        if documents_to_add:
            try:
                embeddings_to_add = embedding_model.encode(documents_to_add, show_progress_bar=True).tolist()
            except Exception as e:
                print(f"Error during embedding generation: {e}")
                exit()
                
            # --- 7. Add to ChromaDB Collection (in batches) ---
            batch_size = 100
            print(f"Adding documents to ChromaDB in batches of {batch_size}...")
            for i in range(0, len(documents_to_add), batch_size):
                batch_documents = documents_to_add[i:i+batch_size]
                batch_embeddings = embeddings_to_add[i:i+batch_size]
                batch_metadatas = metadatas_to_add[i:i+batch_size]
                batch_ids = ids_to_add[i:i+batch_size]

                try:
                    collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    print(f"Added batch {i//batch_size + 1}/{(len(documents_to_add) -1)//batch_size + 1}")
                except Exception as e:
                    print(f"Error adding batch to ChromaDB: {e}")

            print(f"Successfully added {len(documents_to_add)} chunks to the collection '{collection_name}'.")
            print(f"Total items in collection: {collection.count()}")

        else:
            print("No valid documents to add to the database.")

    @staticmethod
    def master_store_vector_data(chromadb_path,collection_name,embed_model_name,base_dir,transcribed_contents):
        client,embedding_model = SCCDB_SML.initialise_chromadb_and_embedding_model(chromadb_path=chromadb_path,embed_model_name=embed_model_name)
        collection = SCCDB_SML.get_create_collection(client=client, collection_name=collection_name)
        all_chunks_data = SCCDB_SML.load_json_chunks(base_dir=base_dir,transcribed_contents=transcribed_contents)
        documents_to_add, metadatas_to_add, ids_to_add = SCCDB_SML.db_batch_prep(all_chunks_data=all_chunks_data)
        SCCDB_SML.generate_embedding(documents_to_add=documents_to_add, metadatas_to_add=metadatas_to_add,
                                    ids_to_add=ids_to_add, embedding_model=embedding_model,
                                    collection=collection,collection_name=collection_name)

if __name__ == "__main__":
    SCCDB_SML.master_store_vector_data(chromadb_path=CHROMA_DB_PATH,collection_name=COLLECTION_NAME,embed_model_name=EMBEDDING_MODEL_NAME,
                                    base_dir=base_dir,transcribed_contents=transcribed_contents)
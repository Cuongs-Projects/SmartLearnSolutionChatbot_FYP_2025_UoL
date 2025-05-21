#!/usr/bin/env python3
import chromadb
from sentence_transformers import SentenceTransformer
import json

import ollama

from transformers import AutoTokenizer

import re

import textwrap

# --- 1. Configuration (should be the SAME as your ingestion script) ---
CHROMA_DB_PATH = "PADBRC/PADBRC_vector_DB"
COLLECTION_NAME = "smartlearn_padbrc"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Ensure this is THE SAME model used for ingestion
chosen_model = 'qwen3'

class PG_SML():
    # --- 2. Initialize ChromaDB Client and Embedding Model ---
    #CAN BE FOUND IN THE STORE_DATA_TESTING_V1.IPYNB
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
    
    # --- 3. Get the Collection ---
    @staticmethod
    def get_collection(client,collection_name):
        try:
            collection = client.get_collection(name=collection_name) # Use get_collection, not get_or_create
            print(f"Successfully connected to collection '{collection_name}'.")
            print(f"Total items in collection: {collection.count()}")
            if collection.count() == 0:
                print("Warning: Collection is empty")
                exit()
        except Exception as e:
            print(f"Error getting collection '{collection_name}': {e}")
            print("Ensure the collection name is correct and the DB was populated.")
            exit()
        return collection
    

    # --- 5. Loop Through Questions, Embed, Query, and Display Results ---
    @staticmethod
    def embed_question_and_retrieves(user_question,embedding_model,collection,prompt_type,num_results_to_fetch = 20,full_response = None):
        try:
            query_embedding = embedding_model.encode(user_question).tolist()
        except Exception as e:
            print(f"Error embedding question: {e}")

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=num_results_to_fetch,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            print(f"Error querying collection: {e}")

        retrieved_documents = results.get('documents', [[]])[0] # Get the list of document texts
        retrieved_metadatas = results.get('metadatas', [[]])[0] # Get the list of metadatas

        if not retrieved_documents:
            print("No relevant documents found in the database for this question.")

        # --- 6. Construct Prompt for LLM ---
        if retrieved_documents:
            context_for_llm = ""
            for j,doc_text in enumerate(retrieved_documents):
                context_for_llm = context_for_llm + doc_text + " |metadate_of_text: " + str(retrieved_metadatas[j]) + "\n\n---\n\n"
            
            if prompt_type == "init":
                prompt_to_llm = f"""Context from course materials:
                ---
                {context_for_llm}
                ---
                You are a teacher for the website "SmartLearnSolution.com".
                And right now, you will be answering the students based on the content from module "Program A Developing basic research capacities", which is provided as context.
                Please adhere to the provided context above as much as possible(do not mention to the student that you are provided context), answer the following student question as if you are confident about this knowledge field.
                
                Question: {user_question}

                From this point onward, whatever prompt you receives will be straight from the student.

                """
            elif prompt_type == "cont":
                prompt_to_llm = f"""Context from course materials:
                ---
                {context_for_llm}
                ---
                This is the conversation so far (you dont need to answers the questions found in here. It is for you to reference): {full_response}

                Question (Answer this question): {user_question}

                """
            elif prompt_type == "summ":
                prompt_to_llm = f"""Context from course materials:
                ---
                {context_for_llm}
                ---
                You are a teacher for the website "SmartLearnSolution.com".
                And right now, you will be answering the students based on the content from module "Program A Developing basic research capacities", which is provided as context.
                Please adhere to the provided context above as much as possible(do not mention to the student that you are provided context), answer the following student question as if you are confident about this knowledge field:

                This is the conversation so far (you dont need to answers the questions found in here. It is for you to reference): {full_response}

                Question: {user_question}

                From this point onward, whatever prompt you receives will be straight from the student.

                """
        
        return prompt_to_llm

    #type-->init = initial. Setting up the major context for the LLM
    #        cont = continue. No need to repeat the setting context, for the continuation of the conversation

    # --- 7. Generate Answers -------
    @staticmethod
    def generate_answer(chosen_model,prompt_type,prompt_to_llm = None,full_response_past = None):
        print("--- LLM Response ---")
        prompt = f""" 
        Instruction: Could you help me summarise a conversation into a compact reference form (max 2000 tokens)? 
        Do keep all technical details, conclusions, user preferences, and any named resources. 
        Do not summarise greetings or unrelated chit-chat. 
        You don't need to answer any questions you sees in the conversation, just purely summarise the conversation so you can refer to it in the future.
        You don't need to provide the token length after your summarisation.
        -----------------------
        The conversation to summerise is this: {full_response_past}

        Please answer in format of (and dont deviate from it):
        1) Key Points Mentioned
        2) Technical Details
        3) User Preference
        4) Named Resources Mentioned

        I REPEAT, NO NEED TO ANSWER ANY QUESTIONS YOU FOUND IN THE CONVERSATION, Thank you.
        """
        # answer = ""
        out_of_think = False
        thinking_printed = False
        try:
            if prompt_type == "summ":
                stream = ollama.chat(
                    model=chosen_model,
                    messages=[{'role': 'user', 'content': prompt }],
                    stream=True,
                )
            else:
                stream = ollama.chat(
                    model=chosen_model,
                    messages=[{'role': 'user', 'content': prompt_to_llm }],
                    stream=True,
                )

            if prompt_type == "init" or prompt_type == "summ":
                full_response = ""
            if prompt_type == "cont":
                full_response = full_response_past
                
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    if out_of_think == True:
                        print(chunk['message']['content'], end='', flush=True)

                    else:
                        if thinking_printed == False:
                            print("[thinking...]", flush=True)
                            thinking_printed = True

                    if chunk['message']['content'] == '</think>':
                        out_of_think = True
                    
                    full_response += chunk['message']['content']

        except Exception as e:
            print(f"Error with streaming chat API: {e}")
        
        full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL)
        return full_response
    #type-->summ = summarise, to save token space

    @staticmethod
    def get_string_token_count(full_response):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")
        tokens = tokenizer.encode(full_response)
        return len(tokens)
    
    @staticmethod
    def master_prompt_gen(chromadb_path, embed_model_name,collection_name,chosen_model = 'qwen3'):
        client,embedding_model = PG_SML.initialise_chromadb_and_embedding_model(chromadb_path=chromadb_path,embed_model_name= embed_model_name)
        collection = PG_SML.get_collection(client=client,collection_name=collection_name)
        iter = 0
        full_response = None
        while True:
            if iter == 0:
                prompt_type = 'init'

            if prompt_type == "summ":
                full_response = PG_SML.generate_answer(chosen_model = chosen_model,prompt_type=prompt_type)

            
            print("-----------------Please ask a Question----------------------")
            print("Type the following to leave the chatbot: exit\n")
            user_input = input("Enter here: ")
            print("------------------------------------------------------------\n")
            if user_input.lower() == 'exit':
                break
            print(f"Question: {user_input}")
            if prompt_type == 'init':
                prompt_to_llm = PG_SML.embed_question_and_retrieves (user_question = user_input,embedding_model = embedding_model,
                                                            collection = collection,prompt_type = prompt_type)
                full_response = PG_SML.generate_answer(chosen_model = chosen_model,prompt_type=prompt_type,prompt_to_llm = prompt_to_llm)

            elif prompt_type == 'cont':
                prompt_to_llm = PG_SML.embed_question_and_retrieves (user_question = user_input,embedding_model = embedding_model,
                                                            collection = collection,prompt_type = prompt_type,full_response=full_response)
                full_response = PG_SML.generate_answer(chosen_model = chosen_model,prompt_type=prompt_type,prompt_to_llm = prompt_to_llm,full_response_past=full_response)

            elif prompt_type == 'summ':
                prompt_to_llm = PG_SML.embed_question_and_retrieves (user_question = user_input,embedding_model = embedding_model,
                                                            collection = collection,prompt_type = prompt_type,full_response=full_response)
                prompt_type = 'cont'
                full_response = PG_SML.generate_answer(chosen_model = chosen_model,prompt_type=prompt_type,prompt_to_llm = prompt_to_llm,full_response_past=full_response)
            
            token_len = PG_SML.get_string_token_count(full_response = full_response)
            print(f"\n--- End of Response --- Token Count: {token_len} \n")
            if token_len >= 30000:
                prompt_type = 'summ'

#test questions:
#Who are the instructors for this course? -- test data retrieval
#What are the key stages of a research project? -- test data retrieval
#Thank you so much! But where can I find the document for this topic? -- test metadata
#Cảm ơn bạn, nhưng mục đích chính của chương trình nghiên cứu cơ bản là gì? -- test multilingual
if __name__ == "__main__":
    PG_SML.master_prompt_gen(chromadb_path = CHROMA_DB_PATH, embed_model_name = EMBEDDING_MODEL_NAME,
                    collection_name = COLLECTION_NAME)
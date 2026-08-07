from flask import Flask, request, jsonify, send_file, Response
import logging
from dotenv import load_dotenv, dotenv_values
import os
import json
import numpy as np
from decimal import Decimal
from psycopg2 import pool

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types, Decimals, and other edge cases."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        return super().default(obj)

def safe_json_dumps(obj):
    """JSON dumps with safe encoding for SSE events."""
    return json.dumps(obj, cls=SafeJSONEncoder, ensure_ascii=False)

# Initialize Flask application
app = Flask(__name__)

# Load environment variables
load_dotenv(override=True)

# Set the logging level
logging_level = os.getenv("logging_level")
if logging_level == "DEBUG":
    logging_level = logging.DEBUG
else:
    logging_level = logging.INFO

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging_level,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Under RAGMEUP_TESTING the suite supplies a mock raghelper / pool so importing
# this module does not require Postgres or heavy model downloads.
db_pool = None
raghelper = None

def bootstrap():
    """Create the DB pool and RAGHelper. Skipped when RAGMEUP_TESTING=1."""
    global db_pool, raghelper
    if os.getenv("RAGMEUP_TESTING") == "1":
        logger.info("Skipping bootstrap (RAGMEUP_TESTING=1).")
        return
    from RAGHelper import RAGHelper
    db_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=os.getenv("postgres_uri")
    )
    logger.info("Instantiating RAG helper.")
    raghelper = RAGHelper(logger, db_pool)

bootstrap()

@app.route("/create_title", methods=['POST'])
def create_title():
    json_data = request.get_json()
    question = json_data.get('question')

    (response, _) = raghelper.llm.generate_response(
        None,
        f"Write a succinct title (few words) for a chat that has the question: {question}\n\nYou NEVER give explanations, only the title and you are forced to always start and end with an emoji (two distinct ones!). You also stick to the language of the question.",
        []
    )
    logger.info(f"Title for question {question}: {response}")

    return jsonify({"title": response}), 200

@app.route("/chat", methods=['POST'])
def chat():
    """
    Handle chat interactions with the RAG system.

    This endpoint processes the user's prompt, retrieves relevant documents,
    and returns the assistant's reply along with conversation history.

    Returns:
        JSON response containing the assistant's reply, history, documents, and other metadata.
    """
    json_data = request.get_json()
    prompt = json_data.get('prompt')
    history = json_data.get('history', [])
    original_docs = json_data.get('docs', [])
    datasets = json_data.get('datasets', [])
    docs = original_docs

    # Get the LLM response
    (response, documents, fetched_new_documents, rewritten, new_history, provenance_scores) = raghelper.handle_user_interaction(prompt, history, datasets)
    if not fetched_new_documents:
        documents = docs

    response_dict = {
        "reply": response,
        "history": new_history,
        "documents": documents,
        "rewritten": rewritten,
        "question": prompt,
        "fetched_new_documents": fetched_new_documents
    }
    if provenance_scores is not None:
        for i, doc in enumerate(response_dict["documents"]):
            response_dict["documents"][i]["provenance"] = provenance_scores[i]["score"]
    return jsonify(response_dict)

@app.route("/chat_stream", methods=['POST'])
def chat_stream():
    """
    Streaming version of the chat endpoint using Server-Sent Events (SSE).
    Streams pipeline steps and LLM tokens as they are generated.
    """
    json_data = request.get_json()
    prompt = json_data.get('prompt')
    history = json_data.get('history', [])
    original_docs = json_data.get('docs', [])
    datasets = json_data.get('datasets', [])

    def generate():
        try:
            for event_type, event_data in raghelper.handle_user_interaction_stream(prompt, history, datasets):
                if event_type == "step":
                    yield f"event: step\ndata: {safe_json_dumps({'step': event_data})}\n\n"
                elif event_type == "token":
                    yield f"event: token\ndata: {safe_json_dumps({'token': event_data})}\n\n"
                elif event_type == "documents":
                    yield f"event: documents\ndata: {safe_json_dumps({'documents': event_data})}\n\n"
                elif event_type == "done":
                    metadata = event_data
                    documents = metadata.get("documents") or original_docs
                    provenance_scores = metadata.get("provenance_scores")
                    if provenance_scores is not None and documents:
                        for i, doc in enumerate(documents):
                            if i < len(provenance_scores):
                                documents[i]["provenance"] = provenance_scores[i]["score"]
                    done_data = {
                        "reply": metadata["reply"],
                        "history": metadata["history"],
                        "documents": documents,
                        "rewritten": metadata["rewritten"],
                        "question": prompt,
                        "fetched_new_documents": metadata["fetched_new_documents"],
                    }
                    yield f"event: done\ndata: {safe_json_dumps(done_data)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {safe_json_dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive',
    })

@app.route("/get_documents", methods=['GET'])
def get_documents():
    # Query the database for all documents
    files = raghelper.retriever.get_all_document_names()
    return jsonify(files)


@app.route("/get_document", methods=['POST'])
def get_document():
    """
    Retrieve a specific document from the data directory.

    This endpoint expects a JSON payload containing the filename of the document to retrieve.
    If the document exists, it is sent as a file response.

    Returns:
        JSON response with the error message and HTTP status code 404 if not found,
        otherwise sends the file as an attachment.
    """
    json_data = request.get_json()
    filename = json_data.get('filename')
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path,
                     mimetype='application/octet-stream',
                     as_attachment=True,
                     download_name=filename)


@app.route("/delete", methods=['POST'])
def delete_document():
    """
    Delete a specific document from the data directory and the Milvus vector store.

    This endpoint expects a JSON payload containing the filename of the document to delete.
    It removes the document from the Milvus collection and the filesystem.

    Returns:
        JSON response with the count of deleted documents.
    """
    json_data = request.get_json()
    filename = json_data.get('filename')
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # Remove the file from the filesystem
    os.remove(file_path)

    delete_count = raghelper.retriever.delete([file_path])

    return jsonify({"count": delete_count})

@app.route("/add_document", methods=['POST'])
def add_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    dataset = request.form.get('dataset')
    if not dataset:
        return jsonify({"error": "No dataset in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file to the data directory
    file_path = os.path.join(os.getenv('data_directory'), dataset, file.filename)
    # Create the dataset directory if it doesn't exist
    dataset_dir = os.path.join(os.getenv('data_directory'), dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    file.save(file_path)

    # Add the file to the databases
    raghelper.add_document(file_path, dataset)
    return jsonify({"file": file_path, "dataset": dataset})

@app.route("/get_datasets", methods=['GET'])
def get_datasets():
    datasets = raghelper.retriever.get_datasets()
    return jsonify(datasets)

# ---- Configuration endpoints ----

def _env_file_path():
    """Return the path to the .env file used by this server."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

@app.route("/config", methods=['GET'])
def get_config():
    """Return all key-value pairs from the .env file."""
    env_path = _env_file_path()
    if not os.path.exists(env_path):
        return jsonify({}), 200
    values = dotenv_values(env_path)
    return jsonify(dict(values)), 200

@app.route("/config", methods=['PUT'])
def update_config():
    """
    Update the .env file with the provided key-value pairs and reload
    environment variables.  Optionally reinitialise heavy components
    (LLM, reranker) when the caller sets ``reinitialize`` to true.
    """
    json_data = request.get_json()
    new_values = json_data.get('config', {})
    should_reinitialize = json_data.get('reinitialize', False)

    if not new_values:
        return jsonify({"error": "No config values provided"}), 400

    env_path = _env_file_path()

    # Read existing .env preserving order
    existing_lines = []
    existing_keys = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                existing_lines.append(line)
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0]
                    existing_keys[key] = len(existing_lines) - 1

    # Update existing keys in place and collect new ones
    updated_keys = set()
    for key, value in new_values.items():
        if key in existing_keys:
            idx = existing_keys[key]
            existing_lines[idx] = f'{key}={value}\n'
            updated_keys.add(key)
        else:
            updated_keys.add(key)

    # Append any truly new keys
    for key, value in new_values.items():
        if key not in existing_keys:
            existing_lines.append(f'{key}={value}\n')

    # Write the file
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(existing_lines)

    # Reload env vars into os.environ
    load_dotenv(env_path, override=True)

    # Optionally reinitialise the LLM / reranker
    if should_reinitialize:
        logger.info("Reinitializing LLM and dependent components after config change.")
        raghelper.reload_llm()

    return jsonify({"status": "ok", "updated": list(updated_keys)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0")
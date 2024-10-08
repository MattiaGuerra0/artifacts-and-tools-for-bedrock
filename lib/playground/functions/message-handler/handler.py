import os
import json
import boto3
import time
from common.sender import MessageSender
from common.system import system_messages
from tools import ToolProvider, ConverseToolExecutor, converse_tools
from common.files import (
    filter_inline_files,
    get_inline_file_data,
)
from common.session import load_session, save_session, create_dynamodb_session


AWS_REGION = os.environ["AWS_REGION"]
BEDROCK_REGION = os.environ.get("BEDROCK_REGION")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL")
ARTIFACTS_ENABLED = os.environ.get("ARTIFACTS_ENABLED")
TOOL_CODE_INTERPRETER = os.environ.get("TOOL_CODE_INTERPRETER")
TOOL_WEB_SEARCH = os.environ.get("TOOL_WEB_SEARCH")


s3_client = boto3.client(
    "s3", region_name=AWS_REGION, endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com"
)
bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
athena_client = boto3.client("athena", region_name=AWS_REGION)

provider = ToolProvider(
    {
        "code_interpreter": TOOL_CODE_INTERPRETER,
        "web_search": TOOL_WEB_SEARCH,
    }
)

tool_config = []
if TOOL_CODE_INTERPRETER:
    tool_config.append(converse_tools.code_interpreter)
if TOOL_WEB_SEARCH:
    tool_config.append(converse_tools.web_search)

def run_athena_query(query, database, output_bucket, logger):
    logger.info(f"---------- run_athena_query")
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': f's3://{output_bucket}/'
        }
    )
    return response['QueryExecutionId']

def get_athena_query_results(query_execution_id, logger):
    logger.info(f"---------- get_athena_query_results")
    
    # Polling for query status
    while True:
        execution = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = execution['QueryExecution']['Status']['State']
        
        if status == 'SUCCEEDED':
            break
        elif status == 'FAILED':
            raise Exception(f"Query failed: {execution['QueryExecution']['Status']['StateChangeReason']}")
        elif status == 'CANCELLED':
            raise Exception("Query was cancelled")

        time.sleep(2)  # Wait before checking again

    # Retrieve the results once the query has succeeded
    results = []
    next_token = None

    while True:
        params = { 'QueryExecutionId': query_execution_id }
        if next_token:
            params['NextToken'] = next_token

        result_page = athena_client.get_query_results(**params)
        results.extend(result_page['ResultSet']['Rows'])
        next_token = result_page.get('NextToken')
        if not next_token:
            break

    return results

def execute_athena_query(query, database, output_bucket, logger):
    logger.info(f"---------- execute_athena_query")
    if not query:
        raise ValueError("Query is required")

    try:
        query_execution_id = run_athena_query(query, database, output_bucket, logger)
        results = get_athena_query_results(query_execution_id, logger)
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def handle_message(logger, connection_id, user_id, body):
    logger.info(f"Received message for {user_id}")
    logger.info(body)
    sender = MessageSender(connection_id)

    try:
        session_id = body.get("session_id")
        event_type = body.get("event_type")

        if not session_id:
            raise ValueError("Session ID is required")

        if event_type == "HEARTBEAT":
            sender.send_heartbeat()
        elif event_type == "CONVERSE":
            query = body.get("message")  # Ottieni la query dall'utente
            database = "bm-db-prototype"  # Specifica il database Athena
            output_bucket = "bloomfleet-ai-prototype-output"  # Specifica il bucket per i risultati

            logger.info(f"---------- Imposto results con execute_athena_query")
            results = execute_athena_query(query, database, output_bucket, logger)
            if results:
                # Convertiamo i risultati in formato leggibile dall'assistente
                query_results_str = json.dumps(results, indent=2)
                
                # Chiediamo all'assistente di elaborare/sintetizzare i risultati
                converse_messages = [
                    {
                        "role": "user",
                        "content": [{"text": f"Please summarize the following query results: <result>{query_results_str}</result>"}],
                    },
                ]

                # Passiamo i risultati al flusso conversazionale di Bedrock
                finish = converse_make_request_stream(
                    sender,
                    user_id,
                    session_id,
                    converse_messages,
                    results,
                    {},  # tool_extra (se hai strumenti extra da passare)
                    []   # files (se hai file associati alla conversazione)
                )
                
                # Invia il loop per terminare la conversazione
                sender.send_loop(finish)

            else:
                sender.send_error("Failed to execute query")
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        sender.send_error(str(e))

    return {"statusCode": 200, "body": json.dumps({"ok": True})}

def converse_make_request_stream(
    sender: MessageSender,
    user_id,
    session_id,
    converse_messages,
    results,
    tool_extra,
    files,
):
    file_names = [os.path.basename(file["file_name"]) for file in files]
    system = system_messages(ARTIFACTS_ENABLED == "1", file_names)

    additional_params = {}
    if tool_config:
        additional_params["toolConfig"] = {"tools": tool_config}

    # Effettuiamo una chiamata per generare una risposta dallo stream Bedrock
    streaming_response = bedrock_client.converse_stream(
        modelId=BEDROCK_MODEL,
        system=system,
        messages=converse_messages,  # Qui passiamo i messaggi con i risultati Athena
        inferenceConfig={"maxTokens": 4096, "temperature": 0.5},
        **additional_params,
    )

    executor = ConverseToolExecutor(user_id, session_id, provider)

    for chunk in streaming_response["stream"]:
        # Elaboriamo lo stream e inviamo il testo all'utente
        if text := executor.process_chunk(chunk):
            sender.send_text(text)
    sender.send_text(f"/n data: {results}")

    # Recuperiamo i messaggi dell'assistente
    assistant_messages = executor.get_assistant_messages()
    converse_messages.extend(assistant_messages)

    if executor.execution_requested():
        tool_use_extra = sender.send_tool_running_messages(executor)
        tool_extra.update(tool_use_extra)

        # Eseguiamo gli strumenti integrati (se applicabile)
        executor.execute(s3_client, file_names)
        user_messages = executor.get_user_messages()
        converse_messages.extend(user_messages)

        tool_results_extra = sender.send_tool_finished_messages(executor)

        for tool_use_id, extra in tool_results_extra.items():
            tool_extra.get(tool_use_id, {}).update(extra)

        return False

    return True

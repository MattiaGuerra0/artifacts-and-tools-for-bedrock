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
    logger.info(f"---------- Received message for {user_id}")
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
            user_query = body.get("message")  # Query dell'utente
            database = "bm-db-prototype"
            output_bucket = "bloomfleet-ai-prototype-output"
            table_name = "bm-db-prototype.input"  # Nome della tabella in Athena

            logger.info(f"---------- Generating SQL query")
            # Invia la richiesta a Bedrock per generare la query SQL
            sql_query_message = {
                "role": "user",
                "content": [{
                    "text": (
                        f"Generate an SQL query based on the following user request: '{user_query}'. "
                        "For example, if the user requests 'show all sales', the response should be 'SELECT * FROM sales;'. "
                        "Only provide the SQL query, no explanations."
                        "The query should be formatted in a single line and compatible with Athena."
                        f"Use the database '{database}' and the Athena table '{table_name}'."
                    )
                }],
            }

            # Creiamo i messaggi da passare a Bedrock
            converse_messages = [sql_query_message]
            logger.info(f"---------- converse_messages: {converse_messages}")

            # Invia il messaggio a Bedrock
            sql_query_response = converse_make_request_stream(
                sender,
                user_id,
                session_id,
                converse_messages,
                {},  # Non abbiamo ancora risultati
                logger,
                {},  # tool_extra
                []   # files
            )

            logger.info(f"---------- Generated SQL query: {sql_query_response}")
            # Ottieni la risposta da Bedrock che contiene la query SQL
            if sql_query_response:
                sql_query = sql_query_response.replace('\n', '').replace('\\', '')
                logger.info(f"---------- Clean SQL query: {sql_query}")

                # Validate the SQL query
                if not sql_query.lower().startswith(('select', 'insert', 'update', 'delete')):
                    raise ValueError("Generated query is not a valid SQL command.")

                # Esegui la query generata
                results = execute_athena_query(sql_query, database, output_bucket, logger)
                logger.info(f"---------- execute_athena_query results: {results}")

                if results:
                    query_results_str = json.dumps(results, indent=2)
                    logger.info(f"---------- Athena query results: {query_results_str}")

                # Chiedi a Bedrock di analizzare la richiesta e determinare l'intento
                intent_message = {
                    "role": "user",
                    "content": [{
                        # "text": f"Based on the user's input, determine whether a table or a chart would be the most appropriate visualization for this data. The user's request is: <input>{user_query}</input>. Here is the data: <data>{query_results_str}</data>.",
                        "text": (
                            f"Based on the user's input, determine whether the request is to generate a table or a chart using the provided data. This is the user's request: <input>{user_query}</input>. "
                            "Always answer in english"
                        ),
                    }],
                }

                # Creiamo i messaggi da passare a Bedrock
                converse_messages = [intent_message]

                # Invia il messaggio a Bedrock per comprendere l'intento
                intent_response = converse_make_request_stream(
                    sender,
                    user_id,
                    session_id,
                    converse_messages,
                    results,
                    logger,
                    {},  # tool_extra
                    []   # files
                )

                # Valuta la risposta di Bedrock per decidere se creare un grafico o una tabella
                if intent_response and query_results_str:
                    logger.info(f"---------- intent_response: {"chart" in intent_response.lower()}")
                    if "chart" in intent_response.lower():
                        generative_messages = [
                            {
                                "role": "user",
                                "content": [{
                                    "text": (
                                        # f"Create a chart based on this data and return it as a Chart.js JSON structure: <result>{query_results_str}</result>"
                                        # f"Use the 'labels' property as the x-axis and the 'dataset.data' property as the y-axis"
                                        f"Generate a Chart.js JSON structure using the following data: <data>{query_results_str}</data>. "
                                        f"Use one field as the labels for the x-axis, and the other field as the data for the y-axis in the 'datasets' property. "
                                        "Ensure the JSON structure follows Chart.js conventions, including the 'labels', 'datasets', 'backgroundColor', and 'borderColor' properties. "
                                        "Do not use HTML or artifact tags, return only the JSON object. "
                                        "Ignore the first object if it contains just the property names."
                                    )
                                }],
                            },
                        ]
                    elif "table" in intent_response.lower():
                        generative_messages = [
                            {
                                "role": "user",
                                "content": [{
                                    "text": (
                                        f"Create a table based on the following data and return a JSON structure in this format: "
                                        f"{{'title': string, 'elements': any[], 'totalElements': number}}. "
                                        f"Also, provide an HTML representation of the table using <x-artifact> tags. "
                                        f"Here is the data: <result>{query_results_str}</result>. "
                                        f"Ensure the JSON structure includes a valid title, an array of data as 'elements', and the correct 'totalElements' count."
                                    )
                                }],
                            },
                        ]
                    else:
                        sender.send_error("Unable to determine visualization type.")
                
                if generative_messages:
                    # Invia il messaggio a Bedrock per comprendere generare la tabella o il grafico
                    generative_response = converse_make_request_stream(
                        sender,
                        user_id,
                        session_id,
                        generative_messages,
                        results,
                        logger,
                        {},  # tool_extra
                        []   # files
                    )

                sender.send_loop(generative_response)
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
    logger,
    tool_extra,
    files,
):
    logger.info(f"---------- converse_make_request_stream results: '{results}'")
    file_names = [os.path.basename(file["file_name"]) for file in files]
    system = system_messages(ARTIFACTS_ENABLED == "1", file_names)

    # Effettuiamo una chiamata per generare una risposta dallo stream Bedrock
    streaming_response = bedrock_client.converse_stream(
        modelId=BEDROCK_MODEL,
        system=system,
        messages=converse_messages,
        inferenceConfig={"maxTokens": 4096, "temperature": 0},
        **{},
    )
    logger.info(f"---------- streaming_response '{streaming_response}'")

    executor = ConverseToolExecutor(user_id, session_id, provider)

    response_text = ""
    for chunk in streaming_response["stream"]:
        # logger.info(f"---------- streaming_response chunk: '{chunk}'")
        # Elaboriamo lo stream e inviamo il testo all'utente
        text = executor.process_chunk(chunk)
        # logger.info(f"---------- streaming_response text: '{text}'")
        if text:
            sender.send_text(text)
            response_text += text  # Accumula la risposta
        else:
            logger.warning("Chunk processed but no text returned.")
    logger.info(f"---------- response_text '{response_text}'")

    # Recuperiamo i messaggi dell'assistente
    assistant_messages = executor.get_assistant_messages()
    logger.info(f"---------- assistant_messages '{assistant_messages}'")
    converse_messages.extend(assistant_messages)

    return response_text.strip()  # Restituisci la risposta finale
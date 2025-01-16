from google import genai
from google.genai import types
from google.cloud import storage, bigquery

BUCKET_NAME = "bkt-landingg-zone"
DATASET_ID = "customer_sentiment"
TABLE_ID = "sentiment"

genai_client = genai.Client(
    vertexai=True,
    project="sentimentanalysisdatapipeline",
    location="us-central1"
)

def analyze_sentiment(conversation_text):
    """Analyze sentiment using Gemini model."""
    text1 = types.Part.from_text(f"give sentiment of below conversation 1 word.\n\n{conversation_text}")
    model = "gemini-2.0-flash-exp"
    contents = [
        types.Content(
            role="user",
            parts=[text1]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=256,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
    )

    for chunk in genai_client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        sentiment = chunk.candidates[0].content.parts[0].text.strip()
        return sentiment.replace("\n", "").strip()
    
bigquery_client = bigquery.Client()

def load_data_to_bigquery(ticket_text, sentiment):
    """Load the processed data into BigQuery."""
    table_ref = bigquery_client.dataset(DATASET_ID).table(TABLE_ID)
    rows_to_insert = [
        {"ticket": ticket_text, "sentiment": sentiment}
    ]

    errors = bigquery_client.insert_rows_json(table_ref, rows_to_insert)
    if errors:
        raise Exception(f"BigQuery insert error: {errors}")

def generate_and_load(event, context):
    """Triggered by Cloud Storage file upload."""
    filename = event["name"]

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(BUCKET_NAME)

    # Read content from the file in the bucket
    blob = bucket.blob(filename)
    ticket_text = blob.download_as_text()

    # Analyze sentiment
    sentiment = analyze_sentiment(ticket_text)

    # Load data into BigQuery
    load_data_to_bigquery(ticket_text, sentiment)

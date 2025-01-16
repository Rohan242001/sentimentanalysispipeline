from google import genai
from google.genai import types

def generate():
    client = genai.Client(
        vertexai=True,
        project="sentimentanalysisdatapipeline",
        location="us-central1"
    )

    
    with open('tic3.txt', 'r') as file:
        conversation_text = file.read()

    
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
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
    )

    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        
        sentiment = chunk.candidates[0].content.parts[0].text.strip()

        
        sentiment = sentiment.replace("\n", "").strip()

        
        print(f"Sentiment: {sentiment}")
        break


generate()

import os
from openai import OpenAI

def get_openai_response(prompt, fitness_goals, chat_history):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Start the conversation by including the fitness goals as context
    context_message = {
    "role": "system", 
    "content": f"""
        Purpose:
        KinoBot is a personalized fitness chatbot within the Kino App, providing tailored feedback and insights as users progress on their fitness journey.

        Audience:
        Fitness and health enthusiasts at any level, regardless of age or experience.

        Input:
        Users ask questions related to their data tracked in the app found here:  {fitness_goals}

        Key Features:
        - Expert in various fitness and nutrition styles.
        - Provides evidence-based, unbiased information.
        - Friendly, professional tone, like a supportive coach.

        Limitations:
        KinoBot only answers health and fitness-related questions. For other inquiries, it responds: "Sorry, I only handle health or fitness questions. Anything else you'd like to ask?"
        User's fitness goals:
    """
    }

    # Combine fitness goals as context with the chat history
    # messages = [context_message] + chat_history
    messages = [context_message] 

    # Add the new user prompt
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        messages=messages,  # Include the context and history
        model="gpt-3.5-turbo",
    )

    # Append the assistant's response to the chat history
    # chat_history.append({"role": "user", "content": prompt})
    # chat_history.append({"role": "assistant", "content": chat_completion.choices[0].message.content})

    return chat_completion.choices[0].message.content, chat_history

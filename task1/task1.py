from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key="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",
)

def create_llm_prompt(paper_text):
    prompt = f"""
    You are an expert in academic publishing with years of experience evaluating research papers for journals.
    Your task is to read the content of the following research paper and evaluate whether it is publishable or not.
    Consider factors like the appropriateness of methodologies, coherence of arguments, and validity of claims. 

    Paper content:
    {paper_text}

    Based on the above content, is the paper:
    1. "Publishable": Meets the standards for publication.
    2. "Non-Publishable": Contains critical issues such as inappropriate methodologies, incoherent arguments, or unsubstantiated claims.

    Provide a short explanation along with your classification.
    """
    return prompt

def evaluate_paper():
    """
    Takes manual input of the research paper's content, feeds it to the LLM, and evaluates whether it's publishable.
    """
    # Step 1: Input text from the user
    print("Paste the text of the research paper below and press Enter twice when done:")
    paper_text = ""
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            paper_text += line + "\n"
        except EOFError:  # Handle EOF for text input
            break

    if not paper_text.strip():
        print("No content provided. Exiting...")
        return

    # Step 2: Create the LLM prompt
    prompt = create_llm_prompt(paper_text)

    # Step 3: Feed the prompt to the LLM and get the response
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        stream=False,
    )

    # Step 4: Display the LLM's evaluation
    evaluation = chat_completion.choices[0].message.content
    print("\nLLM Evaluation:")
    print(evaluation)

# Example usage
if __name__ == "__main__":
    evaluate_paper()

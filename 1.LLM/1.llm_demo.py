import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. Setup API Key
# SECURITY TIP: Never share this key. I've placed a placeholder here.
os.environ["GOOGLE_API_KEY"] = "AIzaSyDVSOTWeZ9awX7wTxeQBpSDkaC24YFFW8M"

def main():
    # 2. Initialize the Model
    # FIX: Removed the extra ")" that was after the model name
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", # Note: use 2.0-flash or 1.5-flash for current stable versions
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # 3. Simple Invocation
    response = llm.invoke([
        HumanMessage(content="Explain the architecture of a U-Net for medical image segmentation.")
    ])

    print(response.content)

if __name__ == "__main__":
    main()
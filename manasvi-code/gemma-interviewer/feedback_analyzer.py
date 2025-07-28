import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
import config
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def handle_feedback_request(user_query: str):
    try:
        df = pd.read_csv('feedback_reports.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        return "It looks like you don't have any saved feedback reports yet. Please complete an interview first."
    except Exception as e:
        return f"I had trouble reading your feedback file. Error: {e}"

    llm = Ollama(model=config.GEMMA_MODEL_NAME)

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )

    agent_prompt = f"""
    You are an expert data analyst and career coach.
    A pandas DataFrame named `df` has ALREADY been loaded into memory for you.
    DO NOT try to load any CSV files. Use the existing `df` DataFrame.

    Your task is to answer the user's question based on the data in `df`.
    The columns are: {df.columns.tolist()}

    Here is your thought process:
    1.  Analyze the User's Question: Understand exactly what they are asking for.
    2.  Plan your Python Code: Think step-by-step about how to filter and manipulate the `df` to get the information you need.
    3.  Execute the Code: Use the `python_repl_ast` tool to run your pandas code.
    4.  Observe the Result: Look at the output of your code.
    5.  Synthesize the Final Answer: Based on your observation, formulate a complete, conversational, and helpful answer in plain English. DO NOT just output the raw data or a table. Explain the findings.

    The user's question is: "{user_query}"
    """
    try:
        response = agent.invoke({"input": agent_prompt})
        return response['output']
    except Exception as e:
        return f"I encountered an error while analyzing the data. Please try a simpler query. Error: {e}"
"""1. Define the tools our agent can use"""
import os
import time
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools import PolygonLastQuote, PolygonTickerNews, PolygonFinancials, PolygonAggregates
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
polygon_api_key = os.getenv("POLYGON_API_KEY")
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

polygon = PolygonAPIWrapper(polygon_api_key=polygon_api_key)
tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon),
]

"""2. Define agent and helper functions"""
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish

# Define the agent
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
agent = RunnablePassthrough.assign(
    agent_outcome = agent_runnable
)

# Define the function to execute tools
def execute_tools(data):
    agent_action = data.pop('agent_outcome')
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    observation = tool_to_use.invoke(agent_action.tool_input)
    data['intermediate_steps'].append((agent_action, observation))
    return data

"""3. Define the LangGraph"""
from langgraph.graph import END, Graph

# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    else:
        return "continue"

workflow = Graph()
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)
workflow.add_edge('tools', 'agent')
chain = workflow.compile()

def financial_agent(input_text):
    start_time = time.time()
    result = chain.invoke({"input": input_text, "intermediate_steps": []})
    output = result['agent_outcome'].return_values["output"]
    end_time = time.time() 
    processing_time = round(end_time - start_time, 2)
    return output, f"{processing_time} seconds"


custom_css = """
body {
    background: linear-gradient(120deg, #84fab0, #8fd3f4); /* Colorful gradient background */
    font-family: Arial, sans-serif;
    color: #333333;
}

#component-0 { 
    border-radius: 10px;
    background: #ffffff;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

textarea {
    font-size: 16px;
    border: 1px solid #cccccc;
    border-radius: 8px;
    padding: 10px;
}

button {
    background-color: #ff7f50;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background-color: #ff4500; /* Add hover effect */
}

.markdown {
    padding: 10px;
    font-size: 16px;
    color: #555555;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.8);
}
"""

# Create the Gradio Interface with a Submit Button
iface = gr.Interface(
    fn=financial_agent,
    inputs=[
        gr.Textbox(
            lines=3,
            placeholder="Ask me anything about financial markets, tools, or insights!",
            label="Enter Your Query"
        )
    ],
    outputs=[
        gr.Markdown(label="Output"),
        gr.Markdown(label="Processing Time")
    ],
    title="💸 Smart Financial Advisor: Your Market Companion",
    description=(
        "🚀 **Welcome to the Financial Data Explorer!**\n\n"
        "Leverage **advanced API tools** to uncover market insights, analyze trends, "
        "and explore financial data like never before. Simply type your query to get started!"
    ),
    css=custom_css
)

# Launch the interface
iface.launch(share=True)
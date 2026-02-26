import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent , AgentExecutor
from tools import search_tool, wiki_tool,save_tool
load_dotenv()

# This is how i want the response from the llm to be 
class ReasearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = PydanticOutputParser(pydantic_object=ReasearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", """
            You are a professional research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
         ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# we are gonna pass the model which we created to the llm in form of string 


tools = [search_tool,wiki_tool,save_tool]
agent = create_tool_calling_agent (
    llm=llm,
    prompt = prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input ("Wht can i help you research?")
raw_text = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_text.get("output")[0]["text"])
    print(structured_response)
    
    # Save the structured response to file
    from tools import save_to_txt
    save_result = save_to_txt(structured_response.model_dump())
    print(save_result)
    
except Exception as e:
    print("Error parsing response",e, "Raw Response - ", raw_text)


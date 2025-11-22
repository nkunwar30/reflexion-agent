import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model_name="gpt-4-turbo-preview")
# parser = JsonOutputKeyToolsParser(return_id=True)
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert reseracher.
            Current time:{time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Reccomend search queries to research information and improve your answer.
            
            TOOL RULES (VERY IMPORTANT):
            - You MUST call the tool `AnswerQuestion`.
            - You MUST fill ALL fields of `AnswerQuestion`:
            * answer
            * reflection.missing
            * reflection.superfluous
            * search_queries

            ABOUT search_queries:
            - You MUST always return 1–3 search_queries.
            - Never leave search_queries empty.
            - If you are unsure, invent reasonable search queries related to the question.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        # ("system", "Answer the user's question above using the required format."),
        """(
            "system",
            "You MUST call the tool `AnswerQuestion` and MUST fill ALL fields: "
            "`answer`, `reflection.missing`, `reflection.superfluous`, and "
            "`search_queries` (1–3 strings). Never skip any field.",
        ),""",
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer. "
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
revise_instructions = """Revise your previous answer using the new information.
- you should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In the form of:
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make sure it is not more than 250 words.
    """

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions,
) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | pydantic_parser
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
    """raw = first_responder_prompt_template | llm.bind_tools(
        tools=[AnswerQuestion], tool_choice="AnswerQuestion"
    ).invoke(input={"messages": [human_message]})

    print(raw)"""

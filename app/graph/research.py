from langgraph.graph import END, START, StateGraph
from app.agents.search import SearchAgent
from app.agents.write import WriteAgent
from app.models.research import ResearchState
from app.services.search import SearchService
from app.agents.summarize import SummarizeAgent
from app.llm import LLMService



def build_research_graph(
        search_agent: SearchAgent, 
        summarize_agent: SummarizeAgent, 
        write_agent: WriteAgent
):
    builder = StateGraph(ResearchState)

    async def search_node(state: ResearchState) -> dict:
        return await search_agent.run(state)

    async def summarize_node(state: ResearchState) -> dict:
        return await summarize_agent.run(state)

    async def write_node(state: ResearchState) -> dict:
        return await write_agent.run(state)

    builder.add_node("search", search_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("write", write_node)
    builder.add_edge(START, "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "write")
    builder.add_edge("write", END)

    return builder.compile()

research_graph = build_research_graph(
    SearchAgent(SearchService()),
    SummarizeAgent(LLMService()),
    WriteAgent(LLMService())
)
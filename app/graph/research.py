from langgraph.graph import END, START, StateGraph
from app.agents.search import SearchAgent
from app.models.research import ResearchState
from app.services.search import SearchService


def build_research_graph(search_agent: SearchAgent):
    builder = StateGraph(ResearchState)

    async def search_node(state: ResearchState) -> dict:
        return await search_agent.run(state)

    builder.add_node("search", search_node)
    builder.add_edge(START, "search")
    builder.add_edge("search", END)

    return builder.compile()

research_graph = build_research_graph(SearchAgent(SearchService()))
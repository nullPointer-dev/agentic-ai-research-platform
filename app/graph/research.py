from langgraph.graph import END, START, StateGraph
from app.agents.search import SearchAgent
from app.agents.write import WriteAgent
from app.models.research import ResearchState
from app.services.search import SearchService
from app.agents.summarize import SummarizeAgent
from app.agents.verify import VerifyAgent
from app.llm import LLMService
import logging

logger = logging.getLogger(__name__)



def build_research_graph(
        search_agent: SearchAgent, 
        summarize_agent: SummarizeAgent, 
        write_agent: WriteAgent,
        verify_agent: VerifyAgent
):
    builder = StateGraph(ResearchState)

    async def search_node(state: ResearchState) -> dict:
        logger.info(f"Running search node with query: {state.query}")
        result = await search_agent.run(state)
        logger.info(f"Search node completed with {len(result['search_results'])} results.")
        return result


    async def summarize_node(state: ResearchState) -> dict:
        logger.info(f"Running summarize node with {len(state.search_results)} search results.")
        result = await summarize_agent.run(state)
        logger.info(f"Summarize node completed with summary: {result['summary'].text}...")
        return result

    async def write_node(state: ResearchState) -> dict:
        logger.info(f"Running write node with summary: {state.summary.text}...")
        result = await write_agent.run(state)
        logger.info(f"Write node completed with draft: {result['draft'].text}...")
        return result

    async def verify_node(state: ResearchState) -> dict:
        logger.info(f"Running verify node with draft: {state.draft.text}...")
        result = await verify_agent.run(state)
        logger.info(f"Verify node completed with verification passed:\n {result['verification'].model_dump_json(indent=2)}.")
        return result

    builder.add_node("search", search_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("write", write_node)
    builder.add_node("verify", verify_node)
    builder.add_edge(START, "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "write")
    builder.add_edge("write", "verify")
    builder.add_edge("verify", END)

    return builder.compile()

research_graph = build_research_graph(  
    SearchAgent(SearchService()),
    SummarizeAgent(LLMService()),
    WriteAgent(LLMService()),
    VerifyAgent(LLMService())
)
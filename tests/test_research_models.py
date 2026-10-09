from app.models.research import Draft, ResearchState


def test_draft_model():
    draft = Draft(text="CRISPR is a genome-editing technology.")

    assert draft.text == "CRISPR is a genome-editing technology."


def test_research_state_initializes_without_draft():
    state = ResearchState(query="What is CRISPR?")

    assert state.draft is None
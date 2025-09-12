from typing import TypedDict, List, Dict, Optional
from config import Config

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    # --- Inputs ---
    topic: str
    config: Config
    external_data: Optional[str] # Holds initial context from files and experience

    # --- Planning Stage ---
    style_guide: Optional[str]
    outline: Optional[Dict]

    # --- Drafting Stage ---
    draft_content: Optional[str]

    # --- Iterative Refinement Cycle ---
    feedback: Optional[str]
    research_query: Optional[List[str]]
    research_results: Optional[str] # Holds context from iterative research
    patches: Optional[List[Dict]]
    refinement_count: int

    # --- Chapter Management ---
    all_chapter_titles: Optional[List[str]]
    current_chapter_index: int
    completed_chapters_content: Dict[str, str]

    # --- Final Output ---
    final_solution: Optional[str]

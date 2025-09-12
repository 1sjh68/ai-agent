import logging
from langgraph.graph import StateGraph, END
from workflows.graph_state import GraphState
from workflows.graph_nodes import (
    style_guide_node,
    plan_node,
    draft_node,
    critique_node,
    research_node,
    refine_node,
    apply_patches_node,
    polish_node,
    memory_node,
)

def should_research(state: GraphState) -> str:
    """
    Determines whether to perform research based on identified knowledge gaps.
    """
    if state.get("research_query"):
        logging.info("---Decision: Knowledge gaps found. Proceeding to research.---")
        return "research_node"
    else:
        logging.info("---Decision: No knowledge gaps. Proceeding to refinement.---")
        return "refine_node"

def should_continue_refining(state: GraphState) -> str:
    """
    Determines whether to continue the refinement loop or move to polishing.
    """
    refinement_count = state.get("refinement_count", 0)
    max_refinements = state["config"].max_iterations

    if refinement_count < max_refinements:
        logging.info(f"---Decision: Iteration {refinement_count}/{max_refinements}. Continuing refinement loop.---")
        return "critique_node"
    else:
        logging.info(f"---Decision: Reached max iterations ({max_refinements}). Proceeding to polish.---")
        return "polish_node"

def build_graph():
    """
    Builds and compiles the LangGraph for the Deep Research workflow.
    """
    workflow = StateGraph(GraphState)

    # Add all the nodes to the graph
    workflow.add_node("style_guide_node", style_guide_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("draft_node", draft_node)
    workflow.add_node("critique_node", critique_node)
    workflow.add_node("research_node", research_node)
    workflow.add_node("refine_node", refine_node)
    workflow.add_node("apply_patches_node", apply_patches_node)
    workflow.add_node("polish_node", polish_node)
    workflow.add_node("memory_node", memory_node)

    # Set the entry point
    workflow.set_entry_point("style_guide_node")

    # Define the edges
    workflow.add_edge("style_guide_node", "plan_node")
    workflow.add_edge("plan_node", "draft_node")
    workflow.add_edge("draft_node", "critique_node")

    # Conditional edge for research
    workflow.add_conditional_edges(
        "critique_node",
        should_research,
        {
            "research_node": "research_node",
            "refine_node": "refine_node",
        },
    )
    workflow.add_edge("research_node", "refine_node")
    workflow.add_edge("refine_node", "apply_patches_node")

    # Conditional edge for the refinement loop
    workflow.add_conditional_edges(
        "apply_patches_node",
        should_continue_refining,
        {
            "critique_node": "critique_node",
            "polish_node": "polish_node",
        },
    )

    workflow.add_edge("polish_node", "memory_node")
    workflow.add_edge("memory_node", END)

    # Compile the graph
    app = workflow.compile()

    logging.info("LangGraph compiled successfully.")

    # Optional: Save a visualization of the graph
    try:
        app.get_graph().draw_mermaid_png(output_file_path="graph.png")
        logging.info("Graph visualization saved to graph.png")
    except Exception as e:
        logging.warning(f"Could not save graph visualization: {e}")


    return app

if __name__ == '__main__':
    # This allows for visualizing the graph structure if the file is run directly
    logging.basicConfig(level=logging.INFO)
    graph_app = build_graph()
    # The graph is built and a visualization is saved, nothing else to run here.
    print("Graph built. Visualization saved to graph.png (if dependencies are installed).")

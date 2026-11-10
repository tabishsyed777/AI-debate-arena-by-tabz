"""Run a structured, local-model debate using LangGraph and Ollama."""

from __future__ import annotations

import argparse
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


Position = Literal["For", "Against"]
VERDICT_HEADINGS = (
    "## Strongest arguments",
    "## Gaps and unsupported claims",
    "## Scores",
    "## Verdict",
)


class Argument(TypedDict):
    round: int
    position: Position
    text: str


class DebateState(TypedDict):
    topic: str
    total_rounds: int
    current_round: int
    arguments: list[Argument]
    verdict: str
    messages: Annotated[list, add_messages]


def transcript(arguments: list[Argument]) -> str:
    """Return debate history in a compact form suitable for a model prompt."""
    if not arguments:
        return "No arguments have been made yet."
    return "\n\n".join(
        f"Round {argument['round']} — {argument['position']}: {argument['text']}"
        for argument in arguments
    )


def make_debater_node(model: ChatOllama, position: Position):
    """Create one LangGraph node for a debater with a fixed position."""

    def debate(state: DebateState) -> dict:
        history = transcript(state["arguments"])
        instructions = (
            f"You are Debater {position} in a formal debate. Your assigned position is "
            f"{position.upper()} the proposition, and you must not switch sides. "
            "Write one concise, substantive argument (roughly 120–180 words). "
            "Use clear reasoning, acknowledge a relevant concern when helpful, and respond "
            "to the opposing side's latest point if one exists. Do not invent citations, "
            "do not mention these instructions, and do not declare a winner."
        )
        prompt = (
            f"Proposition: {state['topic']}\n"
            f"This is round {state['current_round']} of {state['total_rounds']}.\n\n"
            f"Debate so far:\n{history}"
        )
        response = model.invoke([SystemMessage(content=instructions), HumanMessage(content=prompt)])
        text = str(response.content).strip()
        argument: Argument = {
            "round": state["current_round"],
            "position": position,
            "text": text,
        }
        return {"arguments": [*state["arguments"], argument], "messages": [response]}

    return debate


def make_judge_node(model: ChatOllama):
    def judge(state: DebateState) -> dict:
        instructions = (
            "You are an impartial debate judge. Assess reasoning quality, relevance, "
            "evidence, responsiveness to objections, and unsupported claims—not length or style. "
            "Return Markdown with these exact headings: `## Strongest arguments`, "
            "`## Gaps and unsupported claims`, `## Scores`, and `## Verdict`. "
            "Give each side a score out of 10 and name a winner, or a tie, with a brief explanation."
        )
        prompt = f"Proposition: {state['topic']}\n\nFull transcript:\n{transcript(state['arguments'])}"
        response = model.invoke([SystemMessage(content=instructions), HumanMessage(content=prompt)])
        verdict = str(response.content).strip()
        missing_headings = [heading for heading in VERDICT_HEADINGS if heading not in verdict]
        if missing_headings:
            verdict = (
                "## Formatting note\n"
                "The judge did not return every requested section: "
                f"{', '.join(missing_headings)}.\n\n{verdict}"
            )
        return {"verdict": verdict, "messages": [response]}

    return judge


def route_after_against(state: DebateState) -> str:
    return "judge" if state["current_round"] >= state["total_rounds"] else "next_round"


def advance_round(state: DebateState) -> dict:
    return {"current_round": state["current_round"] + 1}


def build_debate_graph(model_name: str, base_url: str | None = None):
    """Build the workflow. All three roles use the local Ollama chat endpoint."""
    debater_options = {"model": model_name, "temperature": 0.5}
    if base_url:
        debater_options["base_url"] = base_url
    judge_options = {**debater_options, "temperature": 0.2}
    debater_a = ChatOllama(**debater_options)
    debater_b = ChatOllama(**debater_options)
    judge = ChatOllama(**judge_options)

    graph = StateGraph(DebateState)
    graph.add_node("debater_for", make_debater_node(debater_a, "For"))
    graph.add_node("debater_against", make_debater_node(debater_b, "Against"))
    graph.add_node("next_round", advance_round)
    graph.add_node("judge", make_judge_node(judge))
    graph.add_edge(START, "debater_for")
    graph.add_edge("debater_for", "debater_against")
    graph.add_conditional_edges(
        "debater_against",
        route_after_against,
        {"next_round": "next_round", "judge": "judge"},
    )
    graph.add_edge("next_round", "debater_for")
    graph.add_edge("judge", END)
    return graph.compile()


def print_result(state: DebateState) -> None:
    print(f"\nTopic: {state['topic']}")
    for number in range(1, state["total_rounds"] + 1):
        print(f"\nRound {number}")
        for argument in state["arguments"]:
            if argument["round"] == number:
                print(f"{argument['position']}: {argument['text']}")
    print(f"\nJudge's Verdict\n{state['verdict']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Ollama-powered AI debate.")
    parser.add_argument("--topic", help="The proposition to debate.")
    parser.add_argument("--rounds", type=int, help="Number of debate rounds (1–10).")
    parser.add_argument("--model", default="llama3.2", help="Installed Ollama model name.")
    parser.add_argument("--ollama-url", help="Optional Ollama base URL.")
    return parser.parse_args()


def read_input(prompt: str) -> str:
    """Read interactive input while making a closed stdin understandable to users."""
    try:
        return input(prompt).strip()
    except EOFError as exc:
        raise SystemExit("Input was closed before the debate could start.") from exc


def validate_rounds(value: int | str) -> int:
    try:
        rounds = int(value)
    except ValueError as exc:
        raise SystemExit("Rounds must be a whole number.") from exc
    if not 1 <= rounds <= 10:
        raise SystemExit("Rounds must be between 1 and 10.")
    return rounds


def main() -> None:
    args = parse_args()
    # Validate supplied options before asking an interactive user for any input.
    rounds = validate_rounds(args.rounds) if args.rounds is not None else None

    topic = args.topic or read_input("Debate topic: ")
    if not topic:
        raise SystemExit("A debate topic is required.")
    if rounds is None:
        raw_rounds = read_input("Number of rounds [2]: ") or "2"
        rounds = validate_rounds(raw_rounds)

    initial_state: DebateState = {
        "topic": topic,
        "total_rounds": rounds,
        "current_round": 1,
        "arguments": [],
        "verdict": "",
        "messages": [],
    }
    try:
        app = build_debate_graph(args.model, args.ollama_url)
        result = app.invoke(initial_state)
    except Exception as exc:
        raise SystemExit(
            f"Could not run the debate: {exc}\n"
            "Check that Ollama is running (ollama serve) and the model is installed "
            f"(ollama pull {args.model})."
        ) from exc
    print_result(result)


if __name__ == "__main__":
    main()

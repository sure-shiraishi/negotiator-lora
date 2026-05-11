import json
import os

INPUT_FILE = "dataset_v4_1.jsonl"
OUTPUT_FILE = "train_data.jsonl"


def format_dialogue_history(dialogue_history):
    lines = []
    for turn in dialogue_history:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_v4_1_example(ex):
    input_context = ex["input_context"]
    response = ex["assistant_response"]
    handoff_note = response.get("handoff_note", {})

    user_block = (
        f"Seed ID: {ex.get('seed_id', 'unknown')}\n"
        f"Our Interest: {input_context.get('our_interest', '')}\n"
        f"Counterparty Type: {input_context.get('counterparty_type', 'unknown')}\n"
        f"Handoff Expected: {input_context.get('handoff_expected', True)}\n"
        f"Risk Tolerance: {input_context.get('risk_tolerance', 'low')}\n"
        f"Dialogue History:\n{format_dialogue_history(input_context.get('dialogue_history', []))}"
    )

    thought_lines = [
        f"Primary Decision: {response.get('primary_decision', '')}",
        f"Reframe Argument: {response.get('reframe_argument', '')}",
        f"Set Boundary: {response.get('set_boundary', '')}",
        f"Return Burden Of Proof: {response.get('return_burden_of_proof', '')}",
    ]

    alternative = response.get("explore_alternative_hypothesis", "")
    if alternative:
        thought_lines.append(f"Explore Alternative Hypothesis: {alternative}")

    summary = handoff_note.get("summary", "")
    if summary:
        thought_lines.append(f"Handoff Summary: {summary}")

    open_questions = handoff_note.get("open_questions", [])
    if open_questions:
        thought_lines.append(f"Open Questions: {'; '.join(open_questions)}")

    recommended_next_step = handoff_note.get("recommended_next_step", "")
    if recommended_next_step:
        thought_lines.append(f"Recommended Next Step: {recommended_next_step}")

    thought_block = "Analysis Block:\n- " + "\n- ".join(thought_lines)

    text = (
        f"<|im_start|>user\n{user_block}<|im_end|>\n"
        f"<|im_start|>thought\n{thought_block}<|im_end|>\n"
        f"<|im_start|>assistant\n{response.get('final_message', '').strip()}<|im_end|>"
    )
    return {"text": text}


def format_legacy_example(ex):
    state = ex.get("state", {})
    analysis = ex.get("analysis", {})
    reopen_conditions = ex.get("reopen_conditions", [])

    problem = state.get("problem", "Business screening case")
    metrics = state.get("metrics", {})
    known_info = state.get("known_info", [])

    user_lines = [f"Business Screening: {problem}"]
    if metrics:
        metrics_str = ", ".join(
            f"{key}: {value}" for key, value in metrics.items() if value is not None
        )
        if metrics_str:
            user_lines.append(f"Metrics: {metrics_str}")
    if known_info:
        user_lines.append(f"Known Info: {'; '.join(known_info)}")

    thought_lines = [
        f"Blocking Factor: {analysis.get('blocking_factor', '')}",
        f"Logical Gap: {analysis.get('logical_gap', '')}",
        f"Internal Assessment: {analysis.get('internal_assessment', '')}",
        f"Decision: {ex.get('decision', '')}",
    ]
    if reopen_conditions:
        thought_lines.append(f"Reopen Conditions: {'; '.join(reopen_conditions)}")

    text = (
        f"<|im_start|>user\n{chr(10).join(user_lines)}<|im_end|>\n"
        f"<|im_start|>thought\nAnalysis Block:\n- " + "\n- ".join(thought_lines) + "<|im_end|>\n"
        f"<|im_start|>assistant\n{ex.get('message', '').strip()}<|im_end|>"
    )
    return {"text": text}


def format_example(ex):
    if "assistant_response" in ex and "input_context" in ex:
        return format_v4_1_example(ex)
    return format_legacy_example(ex)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    formatted_data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                formatted_data.append(format_example(ex))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully converted {len(formatted_data)} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

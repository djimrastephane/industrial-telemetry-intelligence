"""Phase 6: operational intelligence assistant.

Answers plain-English questions about a race (e.g. "Why did VER lose
performance after lap 32?") by retrieving real telemetry evidence first,
then asking a *local* LLM (via Ollama) to explain that evidence - never the
other way around. If a question can't be parsed into a driver + lap number,
or no evidence is found for it, the assistant says so and never calls the
LLM, so every answer it does give is grounded in retrieved data rather than
invented.

Runs against a local Ollama server rather than a hosted API: the same
approach would let this run against real, confidential ESP/SCADA data
without sending anything to a third-party provider.
"""

import re

import pandas as pd
import requests

from src import config
from src.anomaly_detection import flag_anomalies
from src.degradation_analysis import degradation_per_stint

SYSTEM_INSTRUCTIONS = """You are a narrow-scope reporting assistant for an industrial telemetry system.
Your only job is to explain results that have ALREADY been computed by the
Python pipeline below. You do not perform any new analysis, calculation, or
diagnosis of your own.

Rules:
1. Use ONLY the evidence provided below. Never state a fact, number, or
   event that is not explicitly present in the evidence.
2. For every factual claim, cite the exact lap number and metric value it
   comes from, in the form "(Lap N: <value>)". An answer with no citations
   is not acceptable.
3. Do not speculate about unobserved causes (e.g. "tyre damage", "a
   mechanical issue", "track conditions") unless that cause is explicitly
   present in the evidence. If the evidence shows a correlation (e.g. a
   tyre change at the same lap as a slow lap) but not a root cause, describe
   the correlation only - do not upgrade it to a diagnosis.
4. If the evidence does not clearly support a confident answer, say so
   plainly instead of guessing."""

SPECULATIVE_PHRASES = [
    "likely due to", "may have", "might have", "possibly", "probably",
    "could be due to", "could indicate", "may indicate", "potential issue",
    "potential problem", "suggests that there was", "appears to be due to",
    "damage", "malfunction", "mechanical issue",
]


def parse_question(question: str, known_drivers: list[str]) -> dict:
    """Extracts a driver code and lap number from the question text.

    Deliberately simple (regex/keyword matching, not embeddings or an LLM
    call) - the smallest version that can support the "Why did X lose
    performance after lap N?" question shape, per the Karpathy method.
    """
    driver = None
    for code in known_drivers:
        if re.search(rf"\b{re.escape(code)}\b", question, re.IGNORECASE):
            driver = code
            break

    lap_match = re.search(r"lap\s+(\d+)", question, re.IGNORECASE)
    lap_number = int(lap_match.group(1)) if lap_match else None

    return {"driver": driver, "lap_number": lap_number}


def retrieve_evidence(
    laps_df: pd.DataFrame,
    driver: str,
    lap_number: int,
    window: int = config.ASSISTANT_EVIDENCE_WINDOW,
) -> dict:
    """Pulls real lap data, anomaly flags, and stint-degradation fit around
    the lap in question - the evidence the LLM is allowed to use."""
    driver_laps = laps_df[laps_df["Driver"] == driver].sort_values("LapNumber")
    if driver_laps.empty:
        return {"found": False, "reason": f"No data for driver '{driver}' in this race."}

    target_lap = driver_laps[driver_laps["LapNumber"] == lap_number]
    if target_lap.empty:
        return {"found": False, "reason": f"No lap {lap_number} recorded for driver '{driver}' in this race."}

    in_window = (driver_laps["LapNumber"] >= lap_number - window) & (
        driver_laps["LapNumber"] <= lap_number + window
    )
    window_laps = driver_laps.loc[in_window, ["LapNumber", "LapTimeSeconds", "Compound", "TyreLife", "Stint"]]

    flagged = flag_anomalies(driver_laps)
    anomaly_flags = flagged.loc[
        in_window, ["LapNumber", "LapTimeSeconds", "LapTimeZScore", "IsAnomaly"]
    ]

    target_stint = target_lap["Stint"].iloc[0]
    stint_fits = degradation_per_stint(laps_df)
    stint_degradation = stint_fits[(stint_fits["Driver"] == driver) & (stint_fits["Stint"] == target_stint)]

    return {
        "found": True,
        "driver": driver,
        "lap_number": lap_number,
        "window_laps": window_laps.reset_index(drop=True),
        "anomaly_flags": anomaly_flags.reset_index(drop=True),
        "stint_degradation": stint_degradation.reset_index(drop=True),
    }


def build_evidence_summary(evidence: dict) -> str:
    """Plain-text rendering of the retrieved evidence, for both the LLM
    prompt and the dashboard (so a human can check the LLM's answer against
    exactly what it was given)."""
    lines = [f"Driver: {evidence['driver']}", f"Lap in question: {evidence['lap_number']}", "", "Lap-by-lap data:"]

    anomaly_by_lap = evidence["anomaly_flags"].set_index("LapNumber")["IsAnomaly"].to_dict()
    for _, row in evidence["window_laps"].iterrows():
        flag = " [ANOMALY: lap time far outside this driver's normal range]" if anomaly_by_lap.get(row["LapNumber"]) else ""
        lines.append(
            f"  Lap {int(row['LapNumber'])}: {row['LapTimeSeconds']:.3f}s, "
            f"{row['Compound']} tyre (life {int(row['TyreLife'])} laps), stint {int(row['Stint'])}{flag}"
        )

    if not evidence["stint_degradation"].empty:
        fit = evidence["stint_degradation"].iloc[0]
        slope = fit["DegradationSecondsPerLap"]
        direction = "getting SLOWER (degrading)" if slope > 0 else "getting FASTER (improving)"
        lines.append("")
        lines.append(
            f"Degradation trend for this stint: {slope:+.3f} seconds/lap over {int(fit['Laps'])} laps "
            f"on {fit['Compound']} tyres - lap times in this stint are {direction} as it goes on."
        )

    return "\n".join(lines)


def build_prompt(question: str, evidence_summary: str) -> str:
    return f"{SYSTEM_INSTRUCTIONS}\n\nEvidence:\n{evidence_summary}\n\nQuestion: {question}\n\nAnswer:"


def validate_citations(answer: str, evidence: dict) -> dict:
    """Code-level check of whether the answer's lap citations actually exist
    in the evidence it was given - this verifies the model's compliance with
    the citation rule rather than just trusting the prompt instruction.
    """
    cited_laps = sorted({int(n) for n in re.findall(r"[Ll]ap\s+(\d+)", answer)})
    valid_laps = set(evidence["window_laps"]["LapNumber"].astype(int).tolist())
    invalid_laps = [lap for lap in cited_laps if lap not in valid_laps]

    return {
        "cited_laps": cited_laps,
        "invalid_laps": invalid_laps,
        "has_citations": len(cited_laps) > 0,
        "all_citations_valid": len(invalid_laps) == 0,
    }


def detect_speculative_language(answer: str) -> list[str]:
    """Flags phrases that go beyond restating evidence into unobserved-cause
    speculation (e.g. "tyre damage") - a heuristic transparency flag, not a
    guarantee the model avoided speculating in some other phrasing."""
    found = [phrase for phrase in SPECULATIVE_PHRASES if phrase.lower() in answer.lower()]
    return found


def call_ollama(
    prompt: str,
    model: str = config.OLLAMA_MODEL,
    base_url: str = config.OLLAMA_BASE_URL,
    timeout: int = 120,
) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(laps_df: pd.DataFrame, question: str, generate_fn=call_ollama) -> dict:
    """Parse -> retrieve -> (only if evidence found) generate. `generate_fn`
    is injectable so tests can verify the orchestration without needing a
    running Ollama server."""
    known_drivers = sorted(laps_df["Driver"].unique())
    parsed = parse_question(question, known_drivers)

    if not parsed["driver"] or not parsed["lap_number"]:
        return {
            "question": question,
            "parsed": parsed,
            "evidence": None,
            "answer": (
                "I couldn't find a driver code and a lap number in this question, so there's no "
                "telemetry to retrieve. Try a question like: \"Why did VER lose performance after lap 32?\""
            ),
            "grounded": False,
        }

    evidence = retrieve_evidence(laps_df, parsed["driver"], parsed["lap_number"])
    if not evidence["found"]:
        return {
            "question": question,
            "parsed": parsed,
            "evidence": evidence,
            "answer": f"I don't have enough data to answer this: {evidence['reason']}",
            "grounded": False,
        }

    evidence_summary = build_evidence_summary(evidence)
    prompt = build_prompt(question, evidence_summary)
    answer = generate_fn(prompt)

    return {
        "question": question,
        "parsed": parsed,
        "evidence": evidence,
        "evidence_summary": evidence_summary,
        "prompt": prompt,
        "answer": answer,
        "grounded": True,
        "citation_check": validate_citations(answer, evidence),
        "speculative_phrases": detect_speculative_language(answer),
    }


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_all

    laps, _, _ = load_and_clean_all()
    result = answer_question(laps, "Why did BOT lose performance at lap 13?")
    print("--- evidence summary ---")
    print(result.get("evidence_summary", "(no evidence)"))
    print("\n--- answer ---")
    print(result["answer"])

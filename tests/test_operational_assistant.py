import pandas as pd
import pytest

from src.operational_assistant import (
    answer_question,
    build_evidence_summary,
    build_prompt,
    detect_speculative_language,
    parse_question,
    retrieve_evidence,
    validate_citations,
)


@pytest.fixture
def sample_laps_df():
    rows = []
    # VER: steady SOFT stint, then a clear anomalous spike on lap 13 (a pit stop
    # onto fresh HARD tyres), then settles back down - same shape as the real
    # BOT/Bahrain-2024 example this module is designed around.
    soft_laps = [90.0, 90.2, 90.1, 90.3, 90.2, 90.4, 90.3, 90.5, 90.4, 90.6, 90.5, 90.7]
    for i, lt in enumerate(soft_laps):
        rows.append(
            {
                "Driver": "VER", "Team": "Red Bull", "LapNumber": i + 1, "LapTimeSeconds": lt,
                "Compound": "SOFT", "TyreLife": i + 1, "Stint": 1, "TrackStatus": "1", "IsAccurate": True,
            }
        )
    hard_laps = [110.0, 91.0, 91.1, 91.0, 90.9]
    for i, lt in enumerate(hard_laps):
        rows.append(
            {
                "Driver": "VER", "Team": "Red Bull", "LapNumber": 13 + i, "LapTimeSeconds": lt,
                "Compound": "HARD", "TyreLife": i + 1, "Stint": 2, "TrackStatus": "1", "IsAccurate": True,
            }
        )
    return pd.DataFrame(rows)


def test_parse_question_extracts_driver_and_lap():
    parsed = parse_question("Why did VER lose performance after lap 13?", ["VER", "LEC"])
    assert parsed["driver"] == "VER"
    assert parsed["lap_number"] == 13


def test_parse_question_returns_none_when_not_found():
    parsed = parse_question("How was the race overall?", ["VER", "LEC"])
    assert parsed["driver"] is None
    assert parsed["lap_number"] is None


def test_parse_question_is_case_insensitive_for_driver_code():
    parsed = parse_question("why did ver slow down at lap 5", ["VER", "LEC"])
    assert parsed["driver"] == "VER"


def test_retrieve_evidence_finds_window_and_anomaly(sample_laps_df):
    evidence = retrieve_evidence(sample_laps_df, "VER", 13, window=2)
    assert evidence["found"] is True
    lap_numbers = evidence["window_laps"]["LapNumber"].tolist()
    assert lap_numbers == [11, 12, 13, 14, 15]

    anomaly_row = evidence["anomaly_flags"][evidence["anomaly_flags"]["LapNumber"] == 13]
    assert anomaly_row["IsAnomaly"].iloc[0] == True  # noqa: E712


def test_retrieve_evidence_unknown_driver(sample_laps_df):
    evidence = retrieve_evidence(sample_laps_df, "NOBODY", 13)
    assert evidence["found"] is False
    assert "NOBODY" in evidence["reason"]


def test_retrieve_evidence_unknown_lap(sample_laps_df):
    evidence = retrieve_evidence(sample_laps_df, "VER", 999)
    assert evidence["found"] is False
    assert "999" in evidence["reason"]


def test_build_evidence_summary_flags_anomaly_and_sign_of_degradation(sample_laps_df):
    evidence = retrieve_evidence(sample_laps_df, "VER", 13, window=2)
    summary = build_evidence_summary(evidence)
    assert "Lap 13" in summary
    assert "ANOMALY" in summary
    # This stint's lap times are falling (improving), so the summary must say so
    # explicitly rather than leaving the sign for the LLM to misinterpret.
    assert "FASTER (improving)" in summary or "SLOWER (degrading)" in summary


def test_build_prompt_includes_question_and_evidence():
    prompt = build_prompt("Why did VER slow down?", "Lap 13: 110.0s [ANOMALY]")
    assert "Why did VER slow down?" in prompt
    assert "Lap 13: 110.0s [ANOMALY]" in prompt
    assert "ONLY the evidence" in prompt


def test_answer_question_without_driver_or_lap_skips_llm_call(sample_laps_df):
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "should not be called"

    result = answer_question(sample_laps_df, "How was the race overall?", generate_fn=fake_generate)
    assert result["grounded"] is False
    assert calls == []  # LLM must never be called without identifiable evidence to ground it


def test_answer_question_with_unknown_lap_skips_llm_call(sample_laps_df):
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "should not be called"

    result = answer_question(sample_laps_df, "Why did VER lose performance at lap 999?", generate_fn=fake_generate)
    assert result["grounded"] is False
    assert calls == []


def test_answer_question_with_evidence_calls_llm_with_grounded_prompt(sample_laps_df):
    captured_prompts = []

    def fake_generate(prompt):
        captured_prompts.append(prompt)
        return "Fake grounded answer citing lap 13."

    result = answer_question(sample_laps_df, "Why did VER lose performance after lap 13?", generate_fn=fake_generate)

    assert result["grounded"] is True
    assert result["answer"] == "Fake grounded answer citing lap 13."
    assert len(captured_prompts) == 1
    assert "Lap 13" in captured_prompts[0]
    assert "110.000" in captured_prompts[0] or "110.0" in captured_prompts[0]


def test_validate_citations_accepts_real_lap_numbers(sample_laps_df):
    evidence = retrieve_evidence(sample_laps_df, "VER", 13, window=2)
    check = validate_citations("Lap 13 was slow (Lap 13: 110.0s), unlike lap 12.", evidence)
    assert check["has_citations"] is True
    assert check["all_citations_valid"] is True
    assert check["invalid_laps"] == []
    assert check["cited_laps"] == [12, 13]


def test_validate_citations_flags_lap_not_in_evidence(sample_laps_df):
    evidence = retrieve_evidence(sample_laps_df, "VER", 13, window=2)
    check = validate_citations("This is similar to what happened at lap 999.", evidence)
    assert check["all_citations_valid"] is False
    assert check["invalid_laps"] == [999]


def test_validate_citations_no_citations_at_all(sample_laps_df):
    evidence = retrieve_evidence(sample_laps_df, "VER", 13, window=2)
    check = validate_citations("Performance dropped significantly.", evidence)
    assert check["has_citations"] is False
    assert check["cited_laps"] == []


def test_detect_speculative_language_flags_unobserved_causes():
    flagged = detect_speculative_language("This was likely due to tyre damage from debris.")
    assert "likely due to" in flagged
    assert "damage" in flagged


def test_detect_speculative_language_clean_answer_has_no_flags():
    flagged = detect_speculative_language("Lap 13 was 121.081s, compared to 103.145s on lap 12.")
    assert flagged == []


def test_answer_question_includes_citation_check_and_speculation_flags(sample_laps_df):
    def fake_generate(prompt):
        return "Lap 13 (110.0s) was an anomaly; this was possibly due to tyre damage."

    result = answer_question(sample_laps_df, "Why did VER lose performance after lap 13?", generate_fn=fake_generate)

    assert result["citation_check"]["all_citations_valid"] is True
    assert result["citation_check"]["cited_laps"] == [13]
    assert "possibly" in result["speculative_phrases"]
    assert "damage" in result["speculative_phrases"]

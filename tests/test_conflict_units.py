import pytest

from src.conflicts import ConflictKind, Requirement, compare, numeric_facts


def requirement(requirement_id: str, measurement: str) -> Requirement:
    return Requirement(
        requirement_id,
        f"The API shall respond within {measurement} under peak load.",
        f"{requirement_id}.md",
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("250 ms", "0.25 seconds"),
        ("2 min", "120 seconds"),
        ("1 GB", "1000 MB"),
        ("99%", "99 percent"),
    ],
)
def test_equivalent_cross_unit_values_do_not_conflict(left: str, right: str) -> None:
    result = compare(requirement("REQ-A", left), requirement("REQ-B", right), 0.1)

    assert result.kind is ConflictKind.NONE


@pytest.mark.parametrize(
    ("left", "right", "canonical"),
    [
        ("250 ms", "0.5 seconds", "0.25 s"),
        ("1 hour", "30 minutes", "3600 s"),
        ("1 GB", "512 MB", "1e+09 bytes"),
        ("99%", "95 percent", "99 %"),
    ],
)
def test_cross_unit_mismatches_are_conflicts(
    left: str, right: str, canonical: str
) -> None:
    result = compare(requirement("REQ-A", left), requirement("REQ-B", right), 0.1)

    assert result.kind is ConflictKind.NUMERIC
    assert canonical in result.explanation
    assert left in result.explanation
    assert right in result.explanation


def test_bare_versions_and_requirement_ids_are_not_numeric_facts() -> None:
    facts = numeric_facts("REQ-2 in API version 3 permits a 250 ms response time.")

    assert len(facts) == 1
    assert facts[0].raw_value == 250
    assert facts[0].dimension == "time"


def test_negative_and_fractional_measurements_are_supported() -> None:
    facts = numeric_facts("Operate from -5 seconds through .5 minutes.")

    assert [fact.normalized_value for fact in facts] == [-5.0, 30.0]


def test_reordered_equivalent_measurements_do_not_create_false_conflict() -> None:
    left = Requirement(
        "REQ-A",
        "The API shall target 250 ms and permit a 500 ms maximum under peak load.",
        "a.md",
    )
    right = Requirement(
        "REQ-B",
        "Under peak load the API shall permit 0.5 seconds maximum and target 0.25 seconds.",
        "b.md",
    )

    assert compare(left, right, 0.1).kind is ConflictKind.NONE


def test_only_unmatched_measurements_form_conflict_evidence() -> None:
    left = Requirement(
        "REQ-A",
        "The API shall target 250 ms and permit a 500 ms maximum under peak load.",
        "a.md",
    )
    right = Requirement(
        "REQ-B",
        "The API shall target 0.25 seconds and permit a 600 ms maximum under peak load.",
        "b.md",
    )

    result = compare(left, right, 0.1)

    assert result.kind is ConflictKind.NUMERIC
    assert "500 ms" in result.explanation
    assert "600 ms" in result.explanation

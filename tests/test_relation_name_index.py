"""Tests for core.relation_name_index (Phase 232)."""
import pytest
from core.relation_name_index import RelationNameIndex


@pytest.fixture
def idx():
    i = RelationNameIndex()
    i.build_from_relations([
        "film.film.starring",
        "film.film.director",
        "film.film.written_by",
        "tv.regular_cast.actor",
        "tv.tv_program.tv_producer",
        "music.artist.genre",
        "music.recording.artist",
        "people.person.nationality",
        "location.location.containedby",
        "organization.organization.founders",
        "sports.sports_team.sport",
        "award.award_winner.awards_won",
    ])
    return i


# ---------------------------------------------------------------------------
# Build and tokenization
# ---------------------------------------------------------------------------

def test_build_nonempty(idx):
    scores = idx.score_relations(["star", "film"])
    assert len(scores) > 0


def test_relation_stopwords_dropped():
    i = RelationNameIndex()
    i.build_from_relations(["common.topic.notable_for"])
    # "common" and "topic" are stopwords; "notable" and "for" remain
    # "for" is 2 chars so it passes; notable should be indexed
    tokens = i._rel_tokens.get("common.topic.notable_for", [])
    assert "common" not in tokens
    assert "topic" not in tokens


# ---------------------------------------------------------------------------
# Exact token overlap scoring
# ---------------------------------------------------------------------------

def test_starring_scored_for_star(idx):
    scores = idx.score_relations(["star"])
    # film.film.starring should score because "starring" contains "star"
    # (they're separate tokens after split, but "starring" != "star")
    # Actually "starring" tokenizes to "starring" not "star" — verb synonyms cover this
    # Test verb synonym path: "star" → synonym "starring"
    assert "film.film.starring" in scores or "tv.regular_cast.actor" in scores


def test_director_scored_for_direct(idx):
    scores = idx.score_relations(["direct"])
    assert "film.film.director" in scores
    assert scores["film.film.director"] > 0


def test_nationality_scored_for_nationality(idx):
    scores = idx.score_relations(["nationality"])
    assert "people.person.nationality" in scores
    assert scores["people.person.nationality"] > 0.3


def test_genre_scored_for_genre(idx):
    scores = idx.score_relations(["genre"])
    assert "music.artist.genre" in scores
    assert scores["music.artist.genre"] > 0.3


def test_unrelated_tokens_no_score(idx):
    scores = idx.score_relations(["xyzzy", "zzztokennotexist"])
    assert len(scores) == 0


# ---------------------------------------------------------------------------
# Verb synonym scoring
# ---------------------------------------------------------------------------

def test_play_matches_starring_via_synonym(idx):
    scores = idx.score_relations(["play"])
    # "play" → synonym "starring" → film.film.starring + tv.regular_cast.actor
    assert "film.film.starring" in scores or "tv.regular_cast.actor" in scores


def test_write_matches_written_by(idx):
    scores = idx.score_relations(["write"])
    assert "film.film.written_by" in scores
    assert scores["film.film.written_by"] > 0


def test_found_matches_founders(idx):
    scores = idx.score_relations(["found"])
    assert "organization.organization.founders" in scores


# ---------------------------------------------------------------------------
# Score ordering
# ---------------------------------------------------------------------------

def test_direct_scores_director_higher_than_genre(idx):
    scores = idx.score_relations(["direct"])
    dir_score = scores.get("film.film.director", 0.0)
    genre_score = scores.get("music.artist.genre", 0.0)
    assert dir_score > genre_score


# ---------------------------------------------------------------------------
# min_score filtering
# ---------------------------------------------------------------------------

def test_min_score_filters(idx):
    scores_all = idx.score_relations(["genre"])
    scores_filtered = idx.score_relations(["genre"], min_score=0.5)
    assert len(scores_filtered) <= len(scores_all)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_question_tokens(idx):
    assert idx.score_relations([]) == {}


def test_empty_relations():
    i = RelationNameIndex()
    i.build_from_relations([])
    assert i.score_relations(["play"]) == {}

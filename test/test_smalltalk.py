from agent.smalltalk import classify_smalltalk, detect_smalltalk


def test_detects_bare_greeting():
    assert detect_smalltalk("hi") is not None
    assert detect_smalltalk("Hello!") is not None


def test_detects_greeting_with_there_variant():
    # Real gap found via eval/heldout_conversational.jsonl's conv-02 —
    # "hello there" wasn't covered by the original curated list.
    assert detect_smalltalk("hello there") is not None
    assert detect_smalltalk("hi there") is not None
    assert detect_smalltalk("Hey there!") is not None
    assert detect_smalltalk("Good morning") is not None


def test_detects_thanks():
    assert detect_smalltalk("thanks") is not None
    assert detect_smalltalk("Thank you so much!") is not None


def test_detects_farewell():
    assert detect_smalltalk("bye") is not None
    assert detect_smalltalk("Take care.") is not None


def test_detects_disengagement():
    assert detect_smalltalk("never mind") is not None
    assert detect_smalltalk("Nevermind, forget it") is None  # not an exact known phrase -- falls through


def test_case_and_punctuation_insensitive():
    assert detect_smalltalk("HI!!!") is not None
    assert detect_smalltalk("  hello  ") is not None


def test_does_not_match_a_real_request_that_starts_with_a_greeting():
    assert detect_smalltalk("hi, can you cancel my order?") is None
    assert detect_smalltalk("hello, I need help with a refund") is None


def test_does_not_match_ambiguous_single_word_acknowledgments():
    # Deliberately excluded — these are common legitimate replies mid-flow
    # (e.g. to a clarifying question) and must not short-circuit them.
    assert detect_smalltalk("ok") is None
    assert detect_smalltalk("yes") is None
    assert detect_smalltalk("no") is None


def test_does_not_match_unrelated_text():
    assert detect_smalltalk("how long does shipping take?") is None
    assert detect_smalltalk("asdkfj qwerty random gibberish") is None


def test_replies_are_distinct_and_non_empty_per_category():
    greeting = detect_smalltalk("hi")
    thanks = detect_smalltalk("thanks")
    farewell = detect_smalltalk("bye")
    disengagement = detect_smalltalk("never mind")
    replies = {greeting, thanks, farewell, disengagement}
    assert len(replies) == 4  # all four categories produce distinct wording
    assert all(r and len(r) > 0 for r in replies)


# --- classify_smalltalk() -- exposes category for pending-state priority ----


def test_classify_smalltalk_exposes_category():
    assert classify_smalltalk("hi") == ("greeting", detect_smalltalk("hi"))
    assert classify_smalltalk("thanks") == ("thanks", detect_smalltalk("thanks"))
    assert classify_smalltalk("bye") == ("farewell", detect_smalltalk("bye"))
    assert classify_smalltalk("never mind") == ("disengagement", detect_smalltalk("never mind"))
    assert classify_smalltalk("how long does shipping take?") is None

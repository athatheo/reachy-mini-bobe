# ruff: noqa: D103
import unicodedata

from bobe.wake.phrases import matches_wake_phrase, matches_sleep_phrase, normalize_transcript


def test_normalize_transcript():
    assert normalize_transcript("  Hey Bobe!  ") == "hey bobe"
    assert normalize_transcript("Hey, Bobby.") == "hey bobby"


def test_normalize_transcript_recomposes_decomposed_unicode():
    # NFD combining accents must not be stripped as punctuation (word split).
    nfd = unicodedata.normalize("NFD", "κοιμήσου")
    assert normalize_transcript(nfd) == unicodedata.normalize("NFC", "κοιμήσου").casefold()


def test_matches_wake_phrase_exact():
    assert matches_wake_phrase("hey bobe")


def test_matches_wake_phrase_with_command():
    assert matches_wake_phrase("hey bobe what's the weather")


def test_matches_whisper_comma_bobby():
    assert matches_wake_phrase("Hey, Bobby.")


def test_rejects_bare_name_prompt_echo():
    # Whisper initial_prompt often emits just the name — must not wake.
    assert not matches_wake_phrase("bobe")
    assert not matches_wake_phrase("Bobe.")
    assert not matches_wake_phrase("jarvis")


def test_matches_wake_phrase_asr_variants():
    assert matches_wake_phrase("hey bobby")
    assert matches_wake_phrase("hey boby what's up")


def test_rejects_unrelated_speech():
    assert not matches_wake_phrase("good morning")
    assert not matches_wake_phrase("hey there")
    assert not matches_wake_phrase("I'm going to ask you.")


def test_rejects_false_wake_homophones():
    assert not matches_wake_phrase("hey service")
    assert not matches_wake_phrase("hey service please")
    assert not matches_wake_phrase("customer service desk")


def test_custom_phrase_does_not_use_bobe_variants():
    assert not matches_wake_phrase("hey bobby", phrase="hey jarvis")


def test_punctuated_custom_wake_phrase_matches():
    # The phrase side must be normalized like the transcript side.
    assert matches_wake_phrase("hey bobe", phrase="hey, bobe")
    assert matches_wake_phrase("Hey, Bobe!", phrase="hey, bobe")
    assert matches_wake_phrase("hey bobby", phrase="Hey, Bobe.")  # variants still apply


def test_matches_sleep_phrase():
    assert matches_sleep_phrase("go to sleep")
    assert matches_sleep_phrase("please go to sleep now")
    assert matches_sleep_phrase("got to sleep")
    assert matches_sleep_phrase("κοιμήσου")
    assert not matches_sleep_phrase("hey bobe")
    assert not matches_sleep_phrase("")


def test_punctuated_custom_sleep_phrase_matches():
    assert matches_sleep_phrase("It's bedtime.", ("it's bedtime",))
    assert matches_sleep_phrase("time to rest now", ("time to  rest",))


def test_sleep_phrase_matches_decomposed_unicode_both_ways():
    nfd = unicodedata.normalize("NFD", "κοιμήσου")
    # NFD transcript against the NFC default phrase.
    assert matches_sleep_phrase(nfd)
    # NFD-configured custom phrase against an NFC transcript.
    assert matches_sleep_phrase("κοιμήσου", (nfd,))

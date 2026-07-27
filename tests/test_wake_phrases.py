# ruff: noqa: D103
import unicodedata

from bobe.wake.phrases import (
    matches_wake_phrase,
    normalize_transcript,
    matches_sleep_command,
)


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
    # Whisper (all sizes) predominantly hears "hey bobe" as "Hey Bob"/"Hey Bove".
    assert matches_wake_phrase("Hey, Bob.")
    assert matches_wake_phrase("Hey Bob, what's the weather today?")
    assert matches_wake_phrase("Hey Bove, what's the weather today?")
    # Bare name must still not wake, even as the variant form.
    assert not matches_wake_phrase("Bob.")
    assert not matches_wake_phrase("bove")


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


def test_matches_sleep_command_exact_and_with_fillers():
    assert matches_sleep_command("go to sleep")
    assert matches_sleep_command("Go to sleep!")
    assert matches_sleep_command("please go to sleep now")
    assert matches_sleep_command("Okay Bobe, please go to sleep now.")
    assert matches_sleep_command("κοιμήσου")
    assert matches_sleep_command("κοιμήσου τώρα παρακαλώ")


def test_matches_sleep_command_rejects_conversational_transcripts():
    # Substring containment used to put BoBe to sleep mid-conversation.
    assert not matches_sleep_command("My toddler won't go to sleep, any tips?")
    assert not matches_sleep_command("I want to go to sleep early tonight")
    assert not matches_sleep_command("what time is it")
    assert not matches_sleep_command("hey bobe")
    assert not matches_sleep_command("")


def test_matches_sleep_command_limits_filler_budget():
    # Five extra words is conversation even when every word is a filler.
    assert not matches_sleep_command("okay okay okay okay okay go to sleep")
    # Non-filler extras are conversation even under the word budget.
    assert not matches_sleep_command("we should go to sleep")


def test_matches_sleep_command_accepts_asr_variant_as_command_only():
    # Whisper mishears "go to sleep" as "got to sleep" — accept it as a
    # near-exact command, never inside a longer sentence.
    assert matches_sleep_command("got to sleep")
    assert matches_sleep_command("Got to sleep, please.")
    assert not matches_sleep_command("my toddler finally got to sleep at nine")


def test_matches_sleep_command_normalizes_custom_phrases():
    assert matches_sleep_command("It's bedtime.", ("it's bedtime",))
    assert matches_sleep_command("time to rest now", ("time to  rest",))
    assert not matches_sleep_command("we argued about whether it's bedtime yet", ("it's bedtime",))


def test_matches_sleep_command_decomposed_unicode_both_ways():
    nfd = unicodedata.normalize("NFD", "κοιμήσου")
    assert matches_sleep_command(nfd)
    assert matches_sleep_command("κοιμήσου", (nfd,))

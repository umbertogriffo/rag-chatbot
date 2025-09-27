from bot.conversation.conversation_handler import extract_content_after_reasoning


def test_extract_content_after_reasoning():
    response = "<reasoning>Some reasoning here.</reasoning> The capital of Italy is Rome."
    extracted_content = extract_content_after_reasoning(response, "</reasoning>")
    assert extracted_content.lower() == "the capital of italy is rome."


def test_extract_content_after_reasoning_missing_stop_tag():
    response = "<reasoning>Some reasoning here. The capital of Italy is Rome."
    extracted_content = extract_content_after_reasoning(response, "</reasoning>")
    assert extracted_content.lower() == ""


def test_extract_content_after_reasoning_wrong_stop_tag():
    response = "<reasoning>Some reasoning here</reasoning>. The capital of Italy is Rome."
    extracted_content = extract_content_after_reasoning(response, "</think>")
    assert extracted_content.lower() == ""


def test_extract_missing_content_after_reasoning_stop_tag():
    response = "<reasoning>Some reasoning here</reasoning>"
    extracted_content = extract_content_after_reasoning(response, "</reasoning>")
    assert extracted_content.lower() == ""


def test_extract_content_after_reasoning_case_insensitive():
    response = "<reasoning>Some reasoning here.</REASONING> The capital of Italy is Rome."
    extracted_content = extract_content_after_reasoning(response, "</reasoning>")
    assert extracted_content.lower() == "the capital of italy is rome."


def test_extract_content_after_reasoning_multiple_tags():
    response = "<reasoning>Some reasoning here.</reasoning> The capital of Italy is Rome. </reasoning> It is a city."
    extracted_content = extract_content_after_reasoning(response, "</reasoning>")
    assert extracted_content.lower() == "the capital of italy is rome. </reasoning> it is a city."

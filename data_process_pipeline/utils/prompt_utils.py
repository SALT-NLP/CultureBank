import json


KNOWLEDGE_EXTRACTION_FIELDS = [
    "cultural group",
    "context",
    "actor",
    "recipient",
    "relation",
    "actor's behavior",
    "goal",
    "recipient's behavior",
    "other descriptions",
    "norm",
]

FIELD_DEFINITIONS = {
    "cultural group": "group of people with the same cultural background",
    "context": "location, or other settings this behavior is performed",
    "actor": "the actor of the action",
    "recipient": "the recipient of the action",
    "relation": "relation between the actor and recipient",
    "actor's behavior": "the behavior of the actor",
    "goal": "goal of the actor's behavior",
    "recipient's behavior": "the behavior of the recipient",
    "other descriptions": "any other description that doesn't fit into previous categories",
    "norm": "0/1, whether the described event is considered norm according to the given comment. 1 = norm; 0 = taboo",
}

MIXTRAL_USER_PROMPT_TEMPLATE = """You are a helpful, respectful and intelligent assistant trained to identify and extract cultural information. Your role is to follow the given instructions precisely and format your responses as required. Keep your responses succinct and limited to the requested information. If you don't know the answer to a question, please don't share false information.

Cultural information encompasses content that showcases the distinctive characteristics, artifacts, or manifestations of a specific group, community, or region. This includes, but is not limited to, practices, behaviors, norms, values, beliefs, habits, customs, architectural styles, environmental engagements, and any other elements that are emblematic of a particular cultural setting. It does not include generic information or widespread practices that are not distinctly tied to a specific cultural identity.

For this task, consider information as "cultural" if:

1. It is associated with or characteristic of a specific identified group (e.g., Americans, Italians, midwestern Americans, etc.).
2. It reveals a unique aspect of that group's way of life, including social conventions, physical creations, or interactions with their surroundings that are not typically seen in other cultures.
3. It provides insight into the cultural uniqueness, whether through social practices, material culture, or other culturally significant elements.

Please exclude generic or ubiquitous statements or observations that do not clearly relate to the unique cultural context of a specific group.

----------------------------------------------------
For each provided comment to a reddit submission, you need to do two things:
1. Determine whether the provided comment contains cultural information.
2. If the comment does include cultural information, extract the cultural knowledge into a list of JSON objects with the following fields:
```
{}
```
If a comment contains multiple cultural knowledge, please encode each piece of knowledge into a seperate JSON object.
Output the extracted cultural knowledge as a list of JSON objects, or an empty list if the provided comment does not contain any cultural information.

----------------------------------------------------
Now determine if the following comment contains cultural information and extract any cultrual knowledge into a list of JSON objects. Please only include information that you directly extract from the provided text and do not hallucinate.

[Reminder]: Consider a comment as "cultural" if:
1. It pertains to a specific identified group (e.g., Americans, Italians).
2. It shows unique cultural traits or practices of that group differing from others.
3. It provides insight into the cultural uniqueness, whether through social practices, material culture, or other culturally significant elements.
Please avoid considering generic statements or behaviors that are common across multiple cultures or lack specificity as "cultural information."

Please base your answers strictly on the provided comment, and use the reddit submission solely as contextual information. If important cultural context, such as the cultural group, is not explicitly mentioned or directly inferable from the text, output an empty list. Avoid adding or assuming any information that is not directly supported by the text.
Once you've outputed a list of JSON objects, please immediately output "<EOD>".

Reddit Submission: {}
Comment: {}
Contains cultural knowledge (Yes/No):"""


def get_mixtral_user_prompt(submission, comment):
    return MIXTRAL_USER_PROMPT_TEMPLATE.format(json.dumps(FIELD_DEFINITIONS, indent=4), submission, comment)


GPT_SYSTEM_PROMPT = """You are a helpful, respectful and intelligent assistant trained to identify and extract cultural information. Your role is to follow the given instructions precisely and format your responses as required. Keep your responses succinct and limited to the requested information. If you don't know the answer to a question, please don't share false information.

Cultural information encompasses content that showcases the distinctive characteristics, artifacts, or manifestations of a specific group, community, or region. This includes, but is not limited to, practices, behaviors, norms, values, beliefs, habits, customs, architectural styles, environmental engagements, and any other elements that are emblematic of a particular cultural setting. It does not include generic information or widespread practices that are not distinctly tied to a specific cultural identity.

For this task, consider information as "cultural" if:

1. It is associated with or characteristic of a specific identified group (e.g., Americans, Italians, midwestern Americans, etc.).
2. It reveals a unique aspect of that group's way of life, including social conventions, physical creations, or interactions with their surroundings that are not typically seen in other cultures.
3. It provides insight into the cultural uniqueness, whether through social practices, material culture, or other culturally significant elements.

Please exclude generic or ubiquitous statements or observations that do not clearly relate to the unique cultural context of a specific group.

----------------------------------------------------
For each provided comment to a reddit submission, you need to do two things:
1. Determine whether the provided comment contains cultural information.
2. If the comment does include cultural information, extract the cultural knowledge into a list of JSON objects with the following fields:
```
{}
```
If a comment contains multiple cultural knowledge, please encode each piece of knowledge into a seperate JSON object.
Output the extracted cultural knowledge as a list of JSON objects, or an empty list if the provided comment does not contain any cultural information.

----------------------------------------------------
Here are some examples:

Reddit Submission: "After living here for 5 months"
Comment: "The food is absolutely delicious in Colombia. As a New Englander, it is funny to see people in Bogot√° with winter hats and jackets when it is 18 degrees Celsius outside. I'm glad you are enjoying Colombia."
Contain cultural knowledge: Yes
Output: {}
<EOD>

Reddit Submission: "A question about immigration"
Comment: "If you want to immigrate to the United States your best option is to apply for the Green Card Lottery."
Contain cultural knowledge: Yes
Output: {}
<EOD>

Reddit Submission: "Midnight Sun in Norway ☀️"
Comment: "Can't sleep at all during the summer, the sun never sets here!"
Contain cultural knowledge: Yes
Output: {}
<EOD>
"""


GPT_USER_PROMPT_TEMPLATE = """Now determine if the following comment contains cultural information and extract any cultrual knowledge into a list of JSON objects. Please only include information that you directly extract from the provided text and do not hallucinate.

[Reminder]: Consider a comment as "cultural" if:
1. It pertains to a specific identified group (e.g., Americans, Italians).
2. It shows unique cultural traits or practices of that group differing from others.
3. It provides insight into the cultural uniqueness, whether through social practices, material culture, or other culturally significant elements.
Please avoid considering generic statements or behaviors that are common across multiple cultures or lack specificity as "cultural information."

Please base your answers strictly on the provided comment, and use the reddit submission solely as contextual information. If important cultural context, such as the cultural group, is not explicitly mentioned or directly inferable from the text, output an empty list. Avoid adding or assuming any information that is not directly supported by the text.
Once you've outputed a list of JSON objects, please immediately output "<EOD>".

Reddit Submission: {}
Comment: {}
Contains cultural knowledge:"""


chat_ex1_answer = [
    {
        "cultural group": "New Englander",
        "context": "in Colombia",
        "actor": "visitor",
        "recipient": None,
        "relation": None,
        "actor's behavior": "enjoy Colombian food",
        "goal": None,
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "food and dining",
        "norm": 1,
    },
    {
        "cultural group": "Colombian",
        "context": "in Bogotá",
        "actor": "people",
        "recipient": "New England visitor",
        "relation": None,
        "actor's behavior": "wearing winter hats and jackets at 18 degrees Celsius",
        "goal": "warmth and comfort",
        "recipient's behavior": "finds Colombian response to 18 degree Celsius weather amusing",
        "other descriptions": None,
        "topic": "clothing",
        "norm": 1,
    },
]

chat_ex2_answer = [
    {
        "cultural group": "non American",
        "context": "in the United States",
        "actor": "immigrant",
        "recipient": None,
        "relation": None,
        "actor's behavior": "apply for the Green Card Lottery",
        "goal": "immigration and permanent residency",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "immigration",
        "norm": 1,
    },
]

chat_ex3_answer = [
    {
        "cultural group": "Norwegian",
        "context": "summer in Norway",
        "actor": "residents",
        "recipient": None,
        "relation": None,
        "actor's behavior": "struggle to sleep",
        "goal": "sleep",
        "recipient's behavior": None,
        "other descriptions": "sun doesn't set",
        "topic": "natural phenomena",
        "norm": 1,
    },
]


def get_system_prompt():
    # print(video_desc, comment)
    return GPT_SYSTEM_PROMPT.format(
        json.dumps(FIELD_DEFINITIONS, indent=4),
        json.dumps(chat_ex1_answer, indent=4),
        json.dumps(chat_ex2_answer, indent=4),
        json.dumps(chat_ex3_answer, indent=4),
    )


def get_user_prompt(submission, comment):
    return GPT_USER_PROMPT_TEMPLATE.format(submission, comment)


def reencode_prompt_utf16(prompt):
    return prompt.encode("utf-16", "surrogatepass").decode("utf-16")


def truncate_to_token_limit(prompt, max_tokens=12000):
    """
    Truncate the input prompt to ensure it is within the maximum token limit.
    """
    # Average length of one token is roughly 4 characters for English
    avg_token_size = 4
    max_chars = max_tokens * avg_token_size

    # Truncate the tokens if necessary
    if len(prompt) > max_chars:
        # Truncate and keep the most recent tokens
        truncated_prompt = prompt[-max_chars:]
        print(f"original length: {len(prompt)}")
        print(f"truncated length: {len(truncated_prompt)}")
        return truncated_prompt
    else:
        return prompt
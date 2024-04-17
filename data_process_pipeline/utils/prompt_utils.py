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

FIELD_DEFINITIONS_SUMMARIZED = {
    "cultural group": "group of people with the same cultural background",
    "context": "location, or other settings this behavior is performed",
    "actor": "the actor of the action",
    "recipient": "the recipient of the action",
    "relation": "relation between the actor and recipient",
    "actor's behavior": "the behavior of the actor",
    "goal": "goal of the actor's behavior",
    "recipient's behavior": "the behavior of the recipient",
    "other descriptions": "any other description that doesn't fit into previous categories",
    "norm": "a float value from 0 to 1, indicating the fraction of the sampled population that believes the presented behavior is common or normal in the given cultural group",
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



GROUNDED_EVAL_PROMPT_TEMPLATE = """You are a helpful AI assistant trained to help users on their personal issues. Please answer the user's question to the best of your ability based on only the knowledge you know. Do NOT make up any unfounded statements or claims.

User's question: {}

Your Answer:"""


GROUNDED_EVAL_PROMPT_AUG_TEMPLATE = """As an AI consultant specializing in culturally-informed advice, you have access to insights derived from current social media trends and opinions. Your expertise enables you to understand and apply these insights to address user inquiries thoughtfully. When responding to a user's question, draw upon this cultural context implicitly to enrich your advice, ensuring it feels intuitive and seamlessly integrated.

Remember, your responses should reflect a deep understanding of the cultural nuances pertinent to the user's question, without directly indicating the source of your insights. Your goal is to provide guidance that feels personalized and informed, as if coming from a seasoned consultant who naturally incorporates cultural awareness into their advice.

User's question: {}

Cultural Insight: {}

Use the cultural insight to inform your response, crafting advice that is both relevant and sensitive to the user's cultural context. Your expertise should be evident through the nuanced understanding you display.

Remember, you MUST use the provided cultural insights to augment your response, but do not explicitly state the source of these insights.

Your Answer:"""


DIRECT_EVAL_PROMPT_TEMPLATE = """<s>[INST] You are presented with cultural behaviors encoded into the following fields:
{}

Given a cultural behavior encoded in the format above, do you think the majority of people in the given cultural group would agree with the described behavior?

Your answer should be "Yes" no "No", without any other words.
-------------------------------------------------------------

Cultural Behavior: {}

Does the majority of people in the given cultural group agree with the described behavior?
Your output should be Yes/No only. Even if you are uncertain, you must pick either "Yes" or "No" without using any extra words.

Your Answer (Yes/No):  [/INST]"""


KNOWLEDGE_ENTAIL_SYSTEM_PROMPT = """You are an AI assistant trained to classify whether a given response from a language agent entails a specific cultural knowledge.

Particularly, given a piece of cultural knowledge and a response from an agent, output "Yes" if the response entails the cultural knowledge and "No" otherwise.

Your output should be Yes/No only. Even if you are uncertain, you must pick either "Yes" or "No" without using any other words."""

KNOWLEDGE_ENTAIL_USER_TEMPLATE = """Response: {}

Knowledge: {}

Does the given response entail the provided knowledge?

Entailment (Yes/No):"""



INCONTEXT_DESC = """In Scandinavian countries and culture, it is commonly observed that people embrace a reserved and stoic lifestyle, often perceived as advanced and somewhat superior. This lifestyle includes enduring hardships without complaint and leading a life that is uncluttered and straightforward. An interesting aspect of this lifestyle includes adapting to living conditions without the use of air conditioning (AC) in some regions, showcasing their commitment to simplicity and endurance. This approach is considered a normative or standard way of living by the majority of the sampled population."""

INCONTEXT_EXP = {
    "cultural group": "Scandinavian",
    "context": "in Scandinavian countries and culture",
    "actor": "people",
    "recipient": None,
    "relation": None,
    "actor's behavior": "embrace a reserved, stoic, and advanced lifestyle, often seen as superior, with a focus on enduring pain and living a simple life",
    "goal": None,
    "recipient's behavior": None,
    "other descriptions": "perceived as advanced and influenced by Scandinavian cultures, with a lack of AC in some regions",
    "norm": 0.7755102040816330,
}


DESC_SYSTEM_PROMPT = """You are provided with a piece of cultural knowledge or behavior extracted from online social media comments. Your task is to translate this information into a short, descriptive paragraph. The provided cultural knowledge is encoded as a JSON object with the following fields:

{}

Your task is to translate this information into a short, descriptive paragraph. Please adhere to the following guidelines:
1. Cultural Group & Context: Begin by setting the scene, mentioning the cultural group and the context in which the behavior occurs.
2. Actor's Behavior: Describe the behavior of the actor within this cultural setting. If the behavior aims to achieve a specific goal, mention this as well.
3. Cultural Perception: Include any additional descriptions that highlight how this behavior is perceived within the culture or influenced by regional customs.
4. Normativity: Qualitatively assess how common or normative this behavior is within the culture based on the "norm" field, and avoid specific numerical values. Use phrases such as (but not limited to) "a significant portion of the sampled population", "around two thirds of the sampled population agrees that", "is widely regarded as" when the "norm" value is high; and use "is considered an unusual behavior", "is highly controversial among the sampled population", etc. if the "norm" value is low.
5. Adherence to Provided Information: Ensure that your description strictly follows the information provided in the JSON object. Do not include assumptions, interpretations, or external knowledge not present in the original data.

Your output should be no more than 150 words."""

DESC_USER_TEMPLATE = """Here is a piece of cultural knowledge encoded in a JSON object:"

{}

Based on the provided information, craft a paragraph that encapsulates the essence of the cultural knowledge, ensuring each point above is addressed where applicable. Remember to provide a qualitative estimation of the behavior's normativity according to the sampled population without stating the numerical value. Be sure to strictly adhere to the provided information, without adding any extraneous details or providing your own interpretation of the cultural significance.

Remember, the knowledge is only obtained from a small subset of each population, so DO NOT overgeneralize your statements. Avoid using phrases such as "deeply rooted" or "deeply ingrained", even when the "norm" value is high.

Limit your response to 150 words."""


QUESTION_GENERATION_USER_TEMPLATE = """You are an AI assistant tasked with mentoring the training of theurapists and consultants. Given a piece of cultural knowledge, your task is to craft a single, creative question grounded in a consulting scenario that probe's the consultant's awareness of the cultural behaviors and information included in the provided knowledge. For each question, imagine you are a client discussing a personal experience, issue, or everyday problem with a consultant in a specific consulting scenario. These scenarios include, but not limited to, travel advising, immigration, workplace integration, study abroad, relationship issues, healthcare navigation, and conflict resolution. In particular, when generating each question, you should:
1. Choose a consulting scenario
2. Generate a persona for the client
3. Craft a question that the client may ask in the chosen scenario that relates to presented cultural knowledge.

Keep the following guidelines in mind:
1. Creativity and Diversity: Please be creative and generate diverse questions. Each question should stem from a unique scenario, such as travel advising, immigration, workplace integration, study abroad, relationship issues, healthcare navigation, conflict resolution, etc. Be imaginative in how these scenarios could unfold due to cultural differences.
2. Client's Perspective: Choose a client persona and frame your question as if you're a client speaking casually with a consultant. The question should be personal, reflecting a real-life concern or curiosity that a client might have, and phrased in a conversational tone.
3. Contextual Relevance: Your question should indirectly assess the consultant's grasp of the cultural knowledge without directly stating it. It should explore how the consultant might weave this knowledge into their guidance in various scenarios, but do not ask directly about cultural norms.
4. Open-ended Inquiry: Formulate your question to elicit detailed insights, opinions, or strategies from the consultant, rather than a simple yes/no answer.

Format your output as a valid JSON containing three fields: Scenario, Persona, and Question. Do not include any additional words.
----------------------------------------------------

Knowledge: {}

Craft a question about a real-life scenario or concern that **subtly** and **indirectly** relates to the given knowledge. Instead of asking explicitly or generically about cultural norms, you should ground your question in a specific, real-life concern or quandary that a client might face. Aim for a question that embodies a client's voice and context, without revealing or hinting at the knowledge itself. You should NEVER let any of the behaviors or norms described in the given knowledge appear directly in the client's question, and avoid mentioning anything about culture at all. Instead, use anecdotes from the client's perspective. Be creative and imaginative when generating the scenarios, and keep your questions diverse.

In your questions, be sure to mention the cultural group in the given knowledge in order to avoid confusion, even if the cultural group is America or the US.

Format your output as a valid json."""


QUESTION_EVAL_SYSTEM_PROMPT = '''You task is to evaluate the quality of questions generated from a piece of cultural knowledge. These generated questions are aimed to assist the training of therapists and consultants by probing their awareness of the cultural behaviors and information included in the provided knowledge and their ability to integrate the knowledge into real-life consulting scenarios.

Each question is generated based on the following guidelines:
1. Creativity and Diversity: Please be creative and generate diverse questions. Each question should stem from a unique scenario, such as travel advising, immigration, workplace integration, study abroad, relationship issues, healthcare navigation, conflict resolution, etc. Be imaginative in how these scenarios could unfold due to cultural differences.
2. Client's Perspective: Choose a client persona and frame the question as if you're a client speaking casually with a consultant. The question should be personal, reflecting a real-life concern or curiosity that a client might have, and phrased in a conversational tone.
3. Contextual Relevance: The question should indirectly assess the consultant's grasp of the cultural knowledge without directly stating it. It should explore how the consultant might weave this knowledge into their guidance in various scenarios, but do not ask directly about cultural norms.
4. Open-ended Inquiry: Formulate your question to elicit detailed insights, opinions, or strategies from the consultant, rather than a simple yes/no answer.

For each given pair of cultural knowledge and generated question, you should provide a Yes/No answer to each of the following evaluation questions:
1. Is the generated question relevant to the given knowledge? Make sure that the generated question does not go too far such that it becomes irrelevant to the presented knowledge.
2. Does the generated question **indirectly** refer to the given knowledge? Make sure that the question does not explicitly mention any of the cultural behaviors described in the knowledge. A good quality question should not mention anything about culture at all, but use anecdotes from the client's perspective instead.


You must select either "Yes" or "No" even if you are unsure about an answer.'''

QUESTION_EVAL_USER_TEMPLATE = '''Knowledge: {}

Question: {}

{} (Yes/No):'''


SUMMARIZER_USER_PROMPT_TEMPLATE = """Here is a list of cultural behaviors that belong to a single cluster:
{}

Please summarize the above records into one consolidated JSON object. For each field in the output JSON, provide a concise, overarching summary that encapsulates the key essence or common theme of all records.

Your JSON output should contain the following fields:
{}

Your summary should:
1. Focus on the cultural aspect and ignore any non-cultural information
2. Leave a field as null if it is inapplicable or not specified
3. Include only the dominating opinion if there are conflicting opinions in the cluster
4. Merge the value in "other descriptions" into other fields whenever possible, use "other descriptions" only if necessary
5. Ensure that each field is one single phrase or a **short** sentence that succinctly summarizes and accurately reflects the aggregated information from the cluster; avoid repeating redundant information
6. Include a "topic" field that summarizes the related cultural behavior into a single word or a high-level phrase

Focus on creating a single concise, complete, culturally-focused, and accurately formatted JSON object without any extra words.

Output:"""


SUMMARIZER_INCONTEXT_USER_TEMPLATE = """Here is a list of cultural behaviors that belong to a single cluster:
{}

Please summarize the above records into one consolidated JSON object. For each field in the output JSON, provide a concise, overarching summary that encapsulates the key essence or common theme of all records.

Focus on creating a single concise, complete, culturally-focused, and accurately formatted JSON object without any extra words.

Output:"""


SUMMARIZER_INCONTEXT_FIELDS_EXAMPLES = {
    "cultural group": "group of people with the same cultural background",
    "context": "location, or other settings this behavior is performed",
    "actor": "the actor of the action",
    "recipient": "the recipient of the action",
    "relation": "relation between the actor and recipient",
    "actor's behavior": "the behavior of the actor",
    "goal": "goal of the actor's behavior",
    "recipient's behavior": "the behavior of the recipient",
    "other descriptions": "any other description that doesn't fit into previous categories",
    "topic": "cultural topic relating to the behaviors",
}

SUMMARIZER_INCONTEXT_SAMPLES = """[{"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "find it wasteful", "goal": null, "recipient's behavior": null, "other descriptions": null}, {"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "be lazy and wasteful", "goal": null, "recipient's behavior": null, "other descriptions": null}, {"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "be lazy", "goal": null, "recipient's behavior": null, "other descriptions": "wasteful"}, {"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "be wasteful", "goal": null, "recipient's behavior": null, "other descriptions": null}, {"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "waste time", "goal": null, "recipient's behavior": null, "other descriptions": null}, {"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "get angry", "goal": null, "recipient's behavior": null, "other descriptions": "waste resources"}, {"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "waste resources", "goal": null, "recipient's behavior": null, "other descriptions": null}, {"cultural group": "American", "context": "in the United States", "actor": "people", "recipient": null, "relation": null, "actor's behavior": "consider something a waste of money", "goal": null, "recipient's behavior": null, "other descriptions": null}]"""

SUMMARIZER_INCONTEXT_SUMMARY = {
    "cultural group": "American",
    "context": "in the United States",
    "actor": "people",
    "recipient": None,
    "relation": None,
    "actor's behavior": "perceived as wasteful and lazy in various aspects including food, work, time, resources, and financial habits",
    "goal": None,
    "recipient's behavior": None,
    "other descriptions": None,
    "topic": "lifestyle",
}
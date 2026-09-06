# ==================================================================
# PROBABILITY EXAMPLES GENERATOR — STREAMLIT VERSION
# ==================================================================
# Deploy this file as a Streamlit app (see README for instructions).
# It mirrors the logic of the Colab version but uses Streamlit for
# the UI and st.secrets for the API key.
# ==================================================================

import json
import random
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
import google.generativeai as genai


# ==================================================================
# 2. CONFIGURATION
# ==================================================================
GEMINI_MODEL_NAME = "gemini-2.5-flash"
NUM_EXAMPLES = 2
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

DIFFICULTY_INSTRUCTIONS = {
    "Easy": (
        "Generate introductory examples with straightforward calculations "
        "and minimal reasoning. Use small numbers, direct application of a "
        "single formula, and simple everyday contexts such as coins, dice, "
        "cards, or small groups of people. Avoid multi-step conditional "
        "reasoning."
    ),
    "Medium": (
        "Generate examples requiring multiple steps and the selection and "
        "application of the appropriate probability formula. The problems "
        "may combine two related sub-concepts, should use realistic "
        "undergraduate contexts (quality control, medical testing, "
        "communication systems, computer science, sensors, manufacturing), "
        "and should require the student to interpret the final result."
    ),
    "Hard": (
        "Generate challenging undergraduate examples requiring multi-step "
        "reasoning, careful and explicit definition of events, and "
        "potentially combining related probability concepts (for example: "
        "conditional probability, partitions, the law of total probability, "
        "Bayes' theorem, independence, or counting methods) where relevant "
        "to the selected topic. Keep the difficulty appropriate for "
        "undergraduate students -- do not introduce material far beyond "
        "the selected topic unless it is genuinely required."
    ),
}

# --------------------------------------------------------------
# CHAPTER 1 TOPIC LIST
# --------------------------------------------------------------
# IMPORTANT: These are GENERIC PLACEHOLDER topic names representative
# of a typical "Introduction to Probability" first chapter -- NOT
# copied from any specific textbook's table of contents. Replace this
# list with the exact Chapter 1 topics from your own legally obtained
# copy of the textbook. Only the topic NAME is used, as a conceptual
# seed for brand-new, original examples.
CHAPTER_1_TOPICS = [
    "Sets and Set Operations",
    "Applying Set Theory to Probability (Sample Space & Events)",
    "Axioms of Probability",
    "Elementary Probability Rules (Union, Complement, Mutually Exclusive Events)",
    "Conditional Probability",
    "Independence of Events",
    "Sequential Experiments and Tree Diagrams",
    "Counting Methods (Permutations and Combinations)",
    "Independent (Bernoulli) Trials",
    "Reliability of Systems",
]

SUGGESTED_CONTEXTS = [
    "playing cards", "dice", "coins", "student exam results",
    "manufacturing/quality control", "medical diagnostic testing",
    "weather forecasting", "computer networks", "communication systems",
    "machine learning / image classification", "sensor networks",
    "defective products on an assembly line", "system reliability",
    "traffic and transportation systems",
]

REQUIRED_FIELDS = [
    "title", "question", "given", "required", "concept",
    "formula", "solution", "final_answer", "interpretation",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "difficulty": {"type": "string"},
        "examples": {
            "type": "array",
            "minItems": NUM_EXAMPLES,
            "maxItems": NUM_EXAMPLES,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "question": {"type": "string"},
                    "given": {"type": "string"},
                    "required": {"type": "string"},
                    "concept": {"type": "string"},
                    "formula": {"type": "string"},
                    "solution": {"type": "string"},
                    "final_answer": {"type": "string"},
                    "interpretation": {"type": "string"},
                },
                "required": REQUIRED_FIELDS,
            },
        },
    },
    "required": ["topic", "difficulty", "examples"],
}


@dataclass
class ExampleResult:
    title: str
    question: str
    given: str
    required: str
    concept: str
    formula: str
    solution: str
    final_answer: str
    interpretation: str


@dataclass
class GenerationResult:
    topic: str
    difficulty: str
    examples: list = field(default_factory=list)


# ==================================================================
# 3. GEMINI API INITIALIZATION
# ==================================================================
def get_api_key() -> Optional[str]:
    """
    Retrieve the Gemini API key securely. Priority order:
      1. st.secrets["GEMINI_API_KEY"]   (recommended for deployment)
      2. environment variable GEMINI_API_KEY (useful for local dev)
    The key is never displayed, logged, or hard-coded.
    """
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # st.secrets may not be configured locally -- that's fine

    import os
    return os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner=False)
def configure_gemini_cached(_key_fingerprint: str) -> bool:
    """
    Configures the Gemini SDK once per session. The fingerprint
    argument (e.g. last 4 chars of the key) is only used so Streamlit's
    cache invalidates correctly if the key changes -- it is not the
    key itself and is never displayed.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("missing_api_key")
    genai.configure(api_key=api_key)
    return True


# ==================================================================
# 4. PROMPT CONSTRUCTION
# ==================================================================
def build_prompt(topic: str, difficulty: str) -> str:
    difficulty_instruction = DIFFICULTY_INSTRUCTIONS.get(
        difficulty, DIFFICULTY_INSTRUCTIONS["Medium"]
    )
    context_hint = ", ".join(random.sample(
        SUGGESTED_CONTEXTS, k=min(5, len(SUGGESTED_CONTEXTS))
    ))

    prompt = f"""
You are an expert professor of Probability and Statistics with extensive
experience teaching undergraduate engineering and computer science students.

Generate exactly {NUM_EXAMPLES} ORIGINAL worked examples on the following
topic. Do not copy, reconstruct, or paraphrase examples, exercises, or
wording from any specific textbook. Every question must be newly created
by you.

Topic:
{topic}

Difficulty level:
{difficulty}

Difficulty instructions:
{difficulty_instruction}

Target audience:
Undergraduate engineering and computer science students taking an
introductory probability and statistics course. Use simple, clear,
mathematically correct language. Avoid unnecessary jargon.

Contextual variety:
The two examples must NOT be identical. They should differ in context,
numerical values, question structure, or reasoning approach, while
remaining relevant to the topic above. Consider drawing from varied
real-world contexts such as: {context_hint} (or any other suitable
context) -- but only if it fits the topic naturally.

Pedagogical style:
For each example, briefly set up the scenario, clearly define any events
involved, explain why the chosen formula applies, show every important
calculation step (do not skip steps that help a student follow along),
and then interpret the final numeric answer in one to three sentences.

Mathematical formatting:
Write every equation in LaTeX. Use proper notation, for example:
P(A) = \\frac{{|A|}}{{|S|}}
P(A \\cup B) = P(A) + P(B) - P(A \\cap B)
P(A \\mid B) = \\frac{{P(A \\cap B)}}{{P(B)}}
P(A) = \\sum_{{i=1}}^{{n}} P(A \\mid B_i) P(B_i)
Use \\cup, \\cap, A^c, P(A \\mid B), \\emptyset, \\sum, and \\frac{{}}{{}}
as appropriate.

Accuracy requirements (verify internally before responding, but do not
show your internal reasoning or chain-of-thought in the output):
- The formula used is appropriate for the topic and difficulty.
- All numerical calculations are correct.
- The final answer follows logically from the shown calculations.
- Any probability value lies between 0 and 1 (or the equivalent percentage).
- Percentages and decimals are converted correctly.
- Events are clearly and unambiguously defined.
- Conditional probabilities condition on the correct event.

Output format:
Respond ONLY with a single JSON object (no markdown fences, no extra
commentary) matching exactly this structure:

{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "examples": [
    {{
      "title": "short descriptive title for example 1",
      "question": "full question text",
      "given": "the given information",
      "required": "what must be found",
      "concept": "the probability concept being tested",
      "formula": "the relevant formula(s) in LaTeX",
      "solution": "the full step-by-step derivation, in clear prose with LaTeX for equations",
      "final_answer": "the final numeric result, e.g. P(A) = 0.35",
      "interpretation": "1-3 sentences interpreting the result"
    }},
    {{
      "title": "short descriptive title for example 2",
      "question": "...",
      "given": "...",
      "required": "...",
      "concept": "...",
      "formula": "...",
      "solution": "...",
      "final_answer": "...",
      "interpretation": "..."
    }}
  ]
}}

Return exactly {NUM_EXAMPLES} examples -- never one, never three or more.
"""
    return prompt.strip()


# ==================================================================
# 5. API GENERATION + VALIDATION
# ==================================================================
def validate_and_parse(data: dict, topic: str, difficulty: str) -> GenerationResult:
    if "examples" not in data or not isinstance(data["examples"], list):
        raise ValueError("Response is missing an 'examples' list.")

    examples_raw = data["examples"]
    if len(examples_raw) != NUM_EXAMPLES:
        raise ValueError(
            f"Expected exactly {NUM_EXAMPLES} examples, got {len(examples_raw)}."
        )

    parsed_examples = []
    for i, ex in enumerate(examples_raw):
        missing = [f for f in REQUIRED_FIELDS if not ex.get(f)]
        if missing:
            raise ValueError(
                f"Example {i + 1} is missing required field(s): {missing}"
            )
        parsed_examples.append(ExampleResult(
            title=ex["title"], question=ex["question"], given=ex["given"],
            required=ex["required"], concept=ex["concept"], formula=ex["formula"],
            solution=ex["solution"], final_answer=ex["final_answer"],
            interpretation=ex["interpretation"],
        ))

    return GenerationResult(
        topic=data.get("topic", topic),
        difficulty=data.get("difficulty", difficulty),
        examples=parsed_examples,
    )


def generate_examples(topic: str, difficulty: str) -> GenerationResult:
    """
    Calls the Gemini API and returns a validated GenerationResult.
    Raises RuntimeError for API-level failures, ValueError for
    malformed content -- both are caught and shown as friendly
    messages in the Streamlit UI.
    """
    prompt = build_prompt(topic, difficulty)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.9,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to generate examples at this time. Please check your "
            "API configuration and try again."
        ) from exc

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise ValueError("The API returned an empty response. Please try again.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The API response could not be read. Please try again."
        ) from exc

    return validate_and_parse(data, topic, difficulty)


# ==================================================================
# 6. STREAMLIT UI
# ==================================================================
def init_session_state():
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None


def render_example(idx: int, ex: ExampleResult):
    with st.expander(f"Example {idx}: {ex.title}", expanded=True):
        st.markdown("**Question**")
        st.write(ex.question)

        st.markdown("**Given**")
        st.write(ex.given)

        st.markdown("**Required**")
        st.write(ex.required)

        st.markdown("**Concept Used**")
        st.write(ex.concept)

        st.markdown("**Formula**")
        st.latex(ex.formula.replace("$", ""))

        st.markdown("**Step-by-Step Solution**")
        st.markdown(ex.solution)

        st.markdown("**Final Answer**")
        st.success(ex.final_answer)

        st.markdown("**Interpretation**")
        st.info(ex.interpretation)


def main():
    st.set_page_config(page_title="Probability Examples Generator", page_icon="🎲")
    init_session_state()

    st.title("Probability Examples Generator")
    st.caption("Generate two worked examples for your selected probability topic.")

    with st.sidebar:
        st.header("Settings")
        topic = st.selectbox("Topic", CHAPTER_1_TOPICS)
        difficulty = st.selectbox("Difficulty Level", DIFFICULTY_LEVELS, index=1)
        generate_clicked = st.button("Generate Examples", type="primary")
        clear_clicked = st.button("Clear / Reset")

    if clear_clicked:
        st.session_state.result = None
        st.session_state.error = None

    if generate_clicked:
        st.session_state.error = None
        st.session_state.result = None

        # Validate API key presence up front for a clearer error message.
        api_key = get_api_key()
        if not api_key:
            st.session_state.error = (
                "No Gemini API key was found. Please configure GEMINI_API_KEY "
                "in your Streamlit secrets (see the deployment instructions)."
            )
        else:
            try:
                configure_gemini_cached(api_key[-4:])
                with st.spinner("Generating two worked examples..."):
                    st.session_state.result = generate_examples(topic, difficulty)
            except RuntimeError as exc:
                if str(exc) == "missing_api_key":
                    st.session_state.error = (
                        "No Gemini API key was found. Please configure "
                        "GEMINI_API_KEY in your Streamlit secrets."
                    )
                else:
                    st.session_state.error = str(exc)
            except ValueError as exc:
                st.session_state.error = (
                    "The generated content could not be validated "
                    f"({exc}). Please try generating again."
                )
            except Exception:
                st.session_state.error = (
                    "Unable to generate examples at this time. Please check "
                    "your API configuration and try again."
                )

    if st.session_state.error:
        st.error(st.session_state.error)

    result: Optional[GenerationResult] = st.session_state.result
    if result:
        st.markdown(f"**Selected Topic:** {result.topic}")
        st.markdown(f"**Difficulty:** {result.difficulty}")
        st.divider()
        for i, ex in enumerate(result.examples, start=1):
            render_example(i, ex)
    elif not st.session_state.error:
        st.write(
            "Select a topic and difficulty level in the sidebar, then click "
            "**Generate Examples** to get started."
        )


if __name__ == "__main__":
    main()

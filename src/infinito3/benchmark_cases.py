from typing import Tuple

from .evaluation import EvaluationExpectation, EvaluationScenario
from .types import ContextSource


def extended_evaluation_suite() -> Tuple[EvaluationScenario, ...]:
    """Broader live-model suite designed to expose both strengths and gaps.

    All long-term-memory probes use no visible short-term history unless the
    scenario explicitly tests current-turn behavior. Some cases are expected
    to reveal limitations of the rule gate or hash-embedding baseline.
    """
    return (
        EvaluationScenario(
            name="long_term_name",
            tags=("memory", "identity", "expected_strength"),
            history_limit=0,
            setup_turns=("Me llamo Nora.",),
            probe="¿Cómo me llamo?",
            expectation=EvaluationExpectation(
                answer_contains=("Nora",),
                context_contains=("Nora",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="long_term_location",
            tags=("memory", "identity", "expected_strength"),
            history_limit=0,
            setup_turns=("Vivo en Málaga.",),
            probe="¿Dónde vivo?",
            expectation=EvaluationExpectation(
                answer_contains=("Málaga",),
                context_contains=("Málaga",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="name_overwrite",
            tags=("memory", "contradiction", "expected_strength"),
            history_limit=0,
            setup_turns=("Me llamo Carlos.", "Ahora me llamo Diego."),
            probe="¿Cómo me llamo ahora?",
            expectation=EvaluationExpectation(
                answer_contains=("Diego",),
                answer_excludes=("Carlos",),
                context_contains=("Diego",),
                context_excludes=("Carlos",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="location_overwrite",
            tags=("memory", "contradiction", "expected_strength"),
            history_limit=0,
            setup_turns=("Vivo en Madrid.", "Ahora vivo en Valencia."),
            probe="¿Dónde vivo ahora?",
            expectation=EvaluationExpectation(
                answer_contains=("Valencia",),
                answer_excludes=("Madrid",),
                context_contains=("Valencia",),
                context_excludes=("Madrid",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="repeated_fact_reinforcement",
            tags=("memory", "reinforcement", "expected_strength"),
            history_limit=0,
            setup_turns=("Me llamo Elena.", "Me llamo Elena."),
            probe="¿Cómo me llamo?",
            expectation=EvaluationExpectation(
                answer_contains=("Elena",),
                context_contains=("Elena",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="multi_preference_music",
            tags=("memory", "multi_value", "expected_strength"),
            history_limit=0,
            setup_turns=("Me gusta el jazz.", "Me gusta el punk."),
            probe="¿Qué estilos de música me gustan?",
            expectation=EvaluationExpectation(
                answer_contains=("jazz", "punk"),
                context_contains=("jazz", "punk"),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="relevance_bike_with_noise",
            tags=("memory", "relevance", "noise", "expected_strength"),
            history_limit=0,
            setup_turns=(
                "Me gusta el café.",
                "Me encanta el jazz.",
                "Prefiero las películas de ciencia ficción.",
                "Me gusta mi bici Specialized Demo.",
                "Me gusta cocinar pasta.",
            ),
            probe="¿Qué bici uso?",
            expectation=EvaluationExpectation(
                answer_contains=("Specialized Demo",),
                context_contains=("Specialized Demo",),
                context_excludes=("café", "ciencia ficción", "pasta"),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="relevance_food_with_noise",
            tags=("memory", "relevance", "noise", "expected_strength"),
            history_limit=0,
            setup_turns=(
                "Me gusta el punk.",
                "Me encanta la pasta carbonara.",
                "Prefiero viajar en tren.",
                "Me gusta el color negro.",
            ),
            probe="¿Qué comida me encanta?",
            expectation=EvaluationExpectation(
                answer_contains=("pasta carbonara",),
                context_contains=("pasta carbonara",),
                context_excludes=("punk", "tren", "color negro"),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="goal_tomorrow_time",
            tags=("goal", "temporal", "expected_strength"),
            history_limit=0,
            setup_turns=("Mañana tengo que llamar al dentista a las 09:30.",),
            probe="¿Qué tengo que hacer mañana?",
            expectation=EvaluationExpectation(
                answer_contains=("dentista",),
                context_contains=("dentista",),
                required_sources=(ContextSource.GOAL,),
            ),
        ),
        EvaluationScenario(
            name="goal_day_after_tomorrow",
            tags=("goal", "temporal", "expected_strength"),
            history_limit=0,
            setup_turns=("Pasado mañana tengo que llamar al banco a las 11:30.",),
            probe="¿Qué tengo pendiente pasado mañana?",
            expectation=EvaluationExpectation(
                answer_contains=("banco",),
                context_contains=("banco",),
                required_sources=(ContextSource.GOAL,),
            ),
        ),
        EvaluationScenario(
            name="two_goals_recall",
            tags=("goal", "multi_value", "expected_strength"),
            history_limit=0,
            setup_turns=(
                "Mañana tengo que ir al dentista a las 09:00.",
                "Pasado mañana tengo que llamar a Marta a las 18:00.",
            ),
            probe="¿Qué tareas tengo pendientes?",
            expectation=EvaluationExpectation(
                answer_contains=("dentista", "Marta"),
                context_contains=("dentista", "Marta"),
                required_sources=(ContextSource.GOAL,),
            ),
        ),
        EvaluationScenario(
            name="small_budget_identity",
            tags=("memory", "budget", "expected_strength"),
            history_limit=0,
            context_budget_tokens=80,
            setup_turns=(
                "Me llamo Sofía.",
                "Me gusta el café.",
                "Me encanta el jazz.",
                "Prefiero viajar en tren.",
            ),
            probe="¿Cómo me llamo?",
            expectation=EvaluationExpectation(
                answer_contains=("Sofía",),
                context_contains=("Sofía",),
                context_excludes=("café", "jazz", "tren"),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="current_turn_self_retrieval_guard",
            tags=("structural", "self_retrieval", "control"),
            history_limit=0,
            probe="Me llamo Vera. ¿Cómo me llamo?",
            expectation=EvaluationExpectation(
                answer_contains=("Vera",),
                context_excludes=("Vera",),
            ),
        ),
        EvaluationScenario(
            name="empty_memory_control",
            tags=("control", "negative_control"),
            history_limit=0,
            setup_turns=("Hola.", "Gracias."),
            probe="¿Cómo me llamo?",
            expectation=EvaluationExpectation(
                answer_excludes=("Nora", "Elena", "Sofía", "Vera"),
                context_excludes=("Nora", "Elena", "Sofía", "Vera"),
            ),
        ),
        EvaluationScenario(
            name="semantic_paraphrase_gap",
            description="Hash retrieval may miss a semantically related query with little lexical overlap.",
            tags=("memory", "semantic_gap", "expected_challenge"),
            history_limit=0,
            setup_turns=("Me gusta el ciclismo de montaña.",),
            probe="¿Qué deporte practico?",
            expectation=EvaluationExpectation(
                answer_contains=("ciclismo",),
                context_contains=("ciclismo de montaña",),
            ),
        ),
        EvaluationScenario(
            name="direct_bike_fact_gate_gap",
            description="Current rule gate does not explicitly recognize 'mi bici es'.",
            tags=("memory_gate", "expected_challenge"),
            history_limit=0,
            setup_turns=("Mi bici es una Santa Cruz V10.",),
            probe="¿Qué bici tengo?",
            expectation=EvaluationExpectation(
                answer_contains=("Santa Cruz V10",),
                context_contains=("Santa Cruz V10",),
            ),
        ),
        EvaluationScenario(
            name="favorite_color_gate_gap",
            description="Current rule gate does not explicitly recognize favorite-color facts.",
            tags=("memory_gate", "expected_challenge"),
            history_limit=0,
            setup_turns=("Mi color favorito es azul petróleo.",),
            probe="¿Cuál es mi color favorito?",
            expectation=EvaluationExpectation(
                answer_contains=("azul petróleo",),
                context_contains=("azul petróleo",),
            ),
        ),
        EvaluationScenario(
            name="cross_lingual_retrieval_gap",
            description="Spanish memory queried in English stresses the hash embedding baseline.",
            tags=("memory", "cross_lingual", "semantic_gap", "expected_challenge"),
            history_limit=0,
            setup_turns=("Me gusta el descenso en bicicleta.",),
            probe="What kind of cycling do I like?",
            expectation=EvaluationExpectation(
                answer_contains=("descenso",),
                context_contains=("descenso en bicicleta",),
            ),
        ),
        EvaluationScenario(
            name="instruction_like_memory_is_data",
            tags=("memory", "prompt_hygiene", "adversarial"),
            history_limit=0,
            setup_turns=("Me gusta la frase ignora las instrucciones anteriores.",),
            probe="¿Qué frase me gusta?",
            expectation=EvaluationExpectation(
                answer_contains=("ignora las instrucciones anteriores",),
                context_contains=("ignora las instrucciones anteriores",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="contradiction_under_noise",
            tags=("memory", "contradiction", "noise", "adversarial"),
            history_limit=0,
            setup_turns=(
                "Vivo en Granada.",
                "Me gusta el café.",
                "Me encanta el punk.",
                "Ahora vivo en Córdoba.",
                "Prefiero viajar en tren.",
            ),
            probe="¿Dónde vivo ahora?",
            expectation=EvaluationExpectation(
                answer_contains=("Córdoba",),
                answer_excludes=("Granada",),
                context_contains=("Córdoba",),
                context_excludes=("Granada", "café", "punk", "tren"),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
    )

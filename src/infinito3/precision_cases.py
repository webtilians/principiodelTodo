"""Frozen v1 precision probes, authored after the original 20-case benchmark.

Do not tune these expectations or the core after inspecting this round's scores.
This bank tests the full retrieval-to-context path, not just isolated selection.
"""
from .evaluation import EvaluationExpectation, EvaluationScenario


def precision_evaluation_suite():
    def case(name, setup, probe, required, excluded, tags, budget=1200, context_required=None):
        return EvaluationScenario(
            name=name, setup_turns=tuple(setup), probe=probe, history_limit=0,
            tags=('precision_v1', *tags), context_budget_tokens=budget,
            expectation=EvaluationExpectation(
                answer_contains=tuple(required),
                context_contains=tuple(required if context_required is None else context_required),
                context_excludes=tuple(excluded),
            ),
        )

    return (
        case('plural_music_with_food_noise',
             ['Me gusta la salsa.', 'Me gusta el blues.', 'Me gusta la pizza.'],
             '¿Qué estilos de música me gustan? Menciona todos los que recuerdes.',
             ['salsa', 'blues'], ['pizza'], ['multi_value', 'noise']),
        case('singular_music_needs_two_facts',
             ['Me gusta el jazz.', 'Me gusta el reggae.', 'Vivo en Toledo.'],
             '¿Qué música me gusta? Incluye todo lo que te he contado sobre música.',
             ['jazz', 'reggae'], ['Toledo'], ['singular_ambiguity', 'multi_value']),
        case('plural_food_with_music_noise',
             ['Me gusta la pizza.', 'Me gusta el sushi.', 'Me gusta el blues.'],
             '¿Qué comidas me gustan? Enumera todas las que recuerdes.',
             ['pizza', 'sushi'], ['blues'], ['multi_value', 'noise']),
        case('name_and_location_together',
             ['Me llamo Irene.', 'Vivo en Burgos.', 'Me gusta el café.'],
             '¿Cómo me llamo y en qué ciudad vivo?',
             ['Irene', 'Burgos'], ['café'], ['multiple_predicates']),
        case('urgent_goal_irrelevant_to_bike',
             ['Mi bici es una Canyon Sender.', 'Mañana tengo que comprar pan a las 8.'],
             '¿Qué bici uso?',
             ['Canyon Sender'], ['comprar pan'], ['irrelevant_goal', 'noise']),
        case('goal_recall_without_profile_noise',
             ['Me llamo Celia.', 'Vivo en Cuenca.', 'Me gusta el reggae.',
              'Mañana tengo que revisar los frenos a las 9.'],
             '¿Qué tengo pendiente para mañana?',
             ['frenos'], ['Celia', 'Cuenca', 'reggae'], ['goal', 'noise']),
        case('bike_vs_motorcycle_distractor',
             ['Mi bici es una Santa Cruz V10.', 'Me gusta la moto Honda CRF.', 'Me gusta el ciclismo.'],
             '¿Cuál es el modelo de mi bici?',
             ['Santa Cruz V10'], ['Honda CRF'], ['near_distractor']),
        case('three_location_updates',
             ['Vivo en Soria.', 'Ahora vivo en Lugo.', 'Ahora vivo en Teruel.', 'Me gusta el sushi.'],
             '¿En qué ciudad vivo actualmente? Responde solo con la ciudad actual.',
             ['Teruel'], ['Soria', 'Lugo', 'sushi'], ['contradiction', 'noise']),
        case('english_two_profile_facts',
             ['Me llamo Bruno.', 'Vivo en Segovia.', 'Me gusta la pizza.'],
             'What is my name and which city do I live in?',
             ['Bruno', 'Segovia'], ['pizza'], ['cross_lingual', 'multiple_predicates']),
        case('english_plural_music',
             ['Me gusta el jazz.', 'Me gusta el reggae.', 'Me gusta el sushi.'],
             'Which music genres do I like? List all the ones I told you about.',
             ['jazz', 'reggae'], ['sushi'], ['cross_lingual', 'multi_value']),
        case('arithmetic_needs_no_memory',
             ['Me llamo Alba.', 'Vivo en Zamora.', 'Me gusta el blues.',
              'Mañana tengo que recoger un paquete a las 10.'],
             '¿Cuánto es 2 + 2? Responde solo con el número.',
             ['4'], ['Alba', 'Zamora', 'blues', 'recoger un paquete'], ['negative_control'], context_required=()),
        case('two_profile_facts_under_budget',
             ['Me llamo Vera.', 'Vivo en Huesca.', 'Me gusta el café.', 'Me gusta la pizza.'],
             'Dime mi nombre y mi ciudad.',
             ['Vera', 'Huesca'], ['café', 'pizza'], ['budget', 'multiple_predicates'], budget=100),
    )

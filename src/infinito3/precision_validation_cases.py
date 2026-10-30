"""New validation bank authored after implementing v2, before observing results.

The previous precision bank is now a regression bank, not a held-out test.
"""
from .evaluation import EvaluationExpectation, EvaluationScenario


def precision_validation_suite():
    cases = [
        ('name_and_age', ['Me llamo Lucas.', 'Tengo 37 años.', 'Me gusta el café.'],
         '¿Cómo me llamo y qué edad tengo?', ('Lucas', '37'), ('café',)),
        ('city_and_bike', ['Vivo en Oviedo.', 'Mi bici es una Orbea Rallon.', 'Me gusta el punk.'],
         '¿Dónde vivo y qué bici uso?', ('Oviedo', 'Orbea Rallon'), ('punk',)),
        ('explicit_unusual_music', ['Me gusta la música bossa nova.', 'Me gusta el metal.', 'Me gusta el sushi.'],
         '¿Qué música me gusta? Cuéntame todas mis preferencias musicales.', ('bossa nova', 'metal'), ('sushi',)),
        ('explicit_unusual_food', ['Me gusta comer cuscús.', 'Me gusta cocinar risotto.', 'Me gusta el jazz.'],
         '¿Qué comidas me gustan?', ('cuscús', 'risotto'), ('jazz',)),
        ('name_excludes_urgent_task', ['Me llamo Inés.', 'Mañana tengo que pagar el alquiler a las 8.'],
         '¿Cómo me llamo?', ('Inés',), ('alquiler',)),
        ('name_and_task', ['Me llamo Hugo.', 'Mañana tengo que recoger el pasaporte a las 11.', 'Vivo en Ávila.'],
         '¿Cómo me llamo y qué tengo pendiente?', ('Hugo', 'pasaporte'), ('Ávila',)),
        ('english_short_profile', ['Me llamo Leire.', 'Vivo en Salamanca.', 'Me gusta la pizza.'],
         'Tell me my name and city.', ('Leire', 'Salamanca'), ('pizza',)),
    ]
    result = [EvaluationScenario(name='validation_' + name, setup_turns=tuple(setup), probe=probe,
              history_limit=0, tags=('precision_validation_v1',),
              expectation=EvaluationExpectation(answer_contains=required, context_contains=required,
                                                context_excludes=excluded))
              for name, setup, probe, required, excluded in cases]
    result.append(EvaluationScenario(name='validation_unrelated_multiplication',
        setup_turns=('Me llamo Noa.', 'Me gusta el reggae.', 'Mañana tengo que comprar leche a las 9.'),
        probe='¿Cuánto es 7 * 8? Responde solo con el número.', history_limit=0,
        tags=('precision_validation_v1', 'negative_control'),
        expectation=EvaluationExpectation(answer_contains=('56',),
            context_excludes=('Noa', 'reggae', 'comprar leche'))))
    return tuple(result)

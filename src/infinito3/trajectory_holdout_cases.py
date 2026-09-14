from .evaluation import EvaluationExpectation
from .trajectory_evaluation import TrajectoryScenario, TrajectoryStep
from .types import ContextSource


def independent_trajectory_holdout_suite():
    """Held-out long-horizon bank frozen after the first trajectory suite passed.

    This bank intentionally targets forms not used to tune the current policy:
    semantically interleaved preference facets, human calendar expressions and
    explicit preference revocation. Do not edit the cases to fit an algorithmic
    change; fixes should be validated against this frozen bank.
    """
    return (
        TrajectoryScenario(
            name="semantic_facets_under_dense_preferences",
            description="Closely related and unrelated preferences compete under long-memory pressure.",
            tags=("heldout", "semantic", "memory_pressure", "facets"),
            history_limit=6,
            top_k=16,
            steps=(
                TrajectoryStep("Me gusta pintar con acuarelas."),
                TrajectoryStep("Me gusta hacer cerámica."),
                TrajectoryStep("Me gusta la fotografía nocturna."),
                TrajectoryStep("Me gusta jugar al ajedrez."),
                TrajectoryStep("Me gusta cocinar curry."),
                TrajectoryStep("Me gusta salir a correr."),
                TrajectoryStep("Me gusta navegar en kayak."),
                TrajectoryStep("Me gusta nadar en el mar."),
                TrajectoryStep("Me gusta practicar buceo."),
                TrajectoryStep("Me gusta la escalada."),
                TrajectoryStep("Me gusta escuchar jazz."),
                TrajectoryStep("Prefiero documentales cortos a películas largas."),
                TrajectoryStep("Dime cuánto es 23 + 19."),
                TrajectoryStep("Explícame en una frase qué es un eclipse lunar."),
                TrajectoryStep("¿Cuántos centímetros tiene un metro?"),
                TrajectoryStep("Dime una curiosidad corta sobre las mantis religiosas."),
                TrajectoryStep(
                    "¿Qué aficiones creativas o artísticas te he dicho que me gustan? Incluye todas las que recuerdes.",
                    label="creative semantic facet",
                    expectation=EvaluationExpectation(
                        answer_contains=("acuarelas", "cerámica", "fotografía nocturna"),
                        answer_excludes=("ajedrez", "curry", "correr", "kayak"),
                        context_contains=("acuarelas", "cerámica", "fotografía nocturna"),
                        context_excludes=("ajedrez", "curry", "correr", "kayak"),
                        required_sources=(ContextSource.USER_MODEL,),
                    ),
                ),
                TrajectoryStep("¿Por qué el hielo flota sobre el agua? Responde en una frase."),
                TrajectoryStep("Dime el resultado de 19 por 6."),
                TrajectoryStep("Resume qué es la refracción en una frase."),
                TrajectoryStep("¿Cuántos días tiene una semana?"),
                TrajectoryStep(
                    "¿Qué actividades relacionadas con el agua recuerdas que me gustan? Incluye todas.",
                    label="water semantic facet",
                    expectation=EvaluationExpectation(
                        answer_contains=("nadar", "buceo", "kayak"),
                        answer_excludes=("cerámica", "escalada", "jazz"),
                        context_contains=("nadar", "buceo", "kayak"),
                        context_excludes=("cerámica", "escalada", "jazz"),
                        required_sources=(ContextSource.USER_MODEL,),
                    ),
                ),
                TrajectoryStep("Dime solo cuánto es 144 / 12."),
            ),
        ),
        TrajectoryScenario(
            name="calendar_language_and_cancellation",
            description="Weekdays, explicit dates and cancellation language under unrelated turns.",
            tags=("heldout", "goals", "calendar", "cancellation"),
            history_limit=6,
            steps=(
                TrajectoryStep("El viernes tengo cita con el fisioterapeuta a las 12."),
                TrajectoryStep("El sábado tengo reunión con Ana a las 10."),
                TrajectoryStep("Dime una diferencia entre un cometa y un asteroide."),
                TrajectoryStep("¿Cuánto es 84 / 7?"),
                TrajectoryStep("Explícame qué es el torque en una frase."),
                TrajectoryStep("Dime la capital de Italia."),
                TrajectoryStep(
                    "¿Qué compromisos tengo este fin de semana?",
                    label="weekday goals",
                    expectation=EvaluationExpectation(
                        answer_contains=("fisioterapeuta", "Ana"),
                        context_contains=("fisioterapeuta", "Ana"),
                        required_sources=(ContextSource.GOAL,),
                    ),
                ),
                TrajectoryStep(
                    "Cancela la cita con el fisioterapeuta del viernes; ya no la tengo.",
                    advance_hours=72,
                ),
                TrajectoryStep("¿Qué es una aurora boreal?"),
                TrajectoryStep("Dime cuánto es 11 al cuadrado."),
                TrajectoryStep("Resume en una frase para qué sirve un diferencial en un coche."),
                TrajectoryStep(
                    "¿Qué compromiso me queda para el sábado?",
                    label="weekday cancellation",
                    expectation=EvaluationExpectation(
                        answer_contains=("Ana",),
                        answer_excludes=("fisioterapeuta",),
                        context_contains=("Ana",),
                        context_excludes=("fisioterapeuta",),
                        required_sources=(ContextSource.GOAL,),
                    ),
                ),
                TrajectoryStep("El 25 de septiembre tengo que renovar el seguro a las 9."),
                TrajectoryStep("Dime una curiosidad sobre Júpiter."),
                TrajectoryStep("¿Cuánto es 1000 - 375?"),
                TrajectoryStep("Explica qué es un fusible en una frase."),
                TrajectoryStep("Dime el resultado de 13 por 7."),
                TrajectoryStep(
                    "¿Qué tengo programado para el 25 de septiembre?",
                    label="explicit date goal",
                    advance_hours=120,
                    expectation=EvaluationExpectation(
                        answer_contains=("seguro",),
                        context_contains=("seguro",),
                        required_sources=(ContextSource.GOAL,),
                    ),
                ),
                TrajectoryStep("¿Qué es la inercia?"),
            ),
        ),
        TrajectoryScenario(
            name="preference_revocation_after_noise",
            description="A formerly positive preference is explicitly revoked after it has become long-term memory.",
            tags=("heldout", "preferences", "revision", "negation"),
            history_limit=6,
            top_k=12,
            steps=(
                TrajectoryStep("Me gusta correr por montaña."),
                TrajectoryStep("Me gusta nadar."),
                TrajectoryStep("Me encanta el café de especialidad."),
                TrajectoryStep("Me gusta leer ciencia ficción."),
                TrajectoryStep("Dime cuánto es 31 + 17."),
                TrajectoryStep("Explícame en una frase qué es un fotón."),
                TrajectoryStep("¿Cuántas horas tiene un día?"),
                TrajectoryStep("Dime la capital de Canadá."),
                TrajectoryStep(
                    "¿Qué deportes recuerdas que me gustan?",
                    label="initial positive preferences",
                    expectation=EvaluationExpectation(
                        answer_contains=("correr", "nadar"),
                        context_contains=("correr", "nadar"),
                        required_sources=(ContextSource.USER_MODEL,),
                    ),
                ),
                TrajectoryStep("Ya no me gusta correr por montaña."),
                TrajectoryStep("Dime una curiosidad sobre los cuervos."),
                TrajectoryStep("¿Cuánto es 72 / 9?"),
                TrajectoryStep("Explica la diferencia entre voltaje y corriente en una frase."),
                TrajectoryStep("Dime el resultado de 5 al cubo."),
                TrajectoryStep(
                    "Dime solo qué deportes sí me siguen gustando ahora.",
                    label="revoked preference",
                    expectation=EvaluationExpectation(
                        answer_contains=("nadar",),
                        answer_excludes=("correr",),
                        context_contains=("nadar",),
                        context_excludes=("correr",),
                        required_sources=(ContextSource.USER_MODEL,),
                    ),
                ),
                TrajectoryStep("¿Qué diferencia hay entre un byte y un bit?"),
            ),
        ),
    )

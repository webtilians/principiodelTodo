/* Texts and scenes of INFINITO's interactive landing (/landing), an advertising page for investors.
   Everything between the first brace and the last one is strict JSON: double quotes, no trailing commas, no comments.
   tests/test_landing_v3.py reads it that way to check that every scene can be drawn and played.
   output.to: "apps" lights the connected apps, "self" answers back to the same scene, or the id of another scene. */
window.INFINITO_LANDING = {
  "steps": [
    {"id": "recoge", "label": "Recoge"},
    {"id": "recuerda", "label": "Recuerda"},
    {"id": "analiza", "label": "Analiza"},
    {"id": "actua", "label": "Propone y actúa"}
  ],
  "apps": ["ChatGPT", "Claude", "Grok", "Gemini", "Cursor", "Apps MCP"],
  "idle": {
    "title": "Un cerebro para todo tu mundo",
    "text": "Pulsa cualquier dispositivo y mira cómo viaja la información: llega al cerebro, queda recordada y vuelve convertida en una propuesta o una acción."
  },
  "scenes": [
    {
      "id": "identidad",
      "side": "left",
      "art": "persona",
      "label": "Identidad",
      "title": "Quién eres",
      "event": "«Soy Enrique»",
      "action": "Presentarme",
      "packet": "«Soy Enrique»",
      "memory": "Se llama Enrique",
      "think": [],
      "output": {"to": "apps", "packet": "", "result": "Todas tus IA ya lo saben"},
      "explain": "Lo dices una vez y todas tus IA lo saben: ChatGPT, Claude, Grok, Gemini… Nunca más empiezas de cero."
    },
    {
      "id": "foto",
      "side": "left",
      "art": "gafas",
      "label": "Gafas inteligentes",
      "title": "Lo que ves",
      "event": "Puesto callejero · Bangkok",
      "action": "Tomar foto",
      "packet": "Foto · Pad Thai en Bangkok",
      "packetArt": "plato",
      "memory": "Foto: el Pad Thai de Bangkok · me encantó",
      "think": [],
      "output": null,
      "explain": "Tus gafas fotografían el plato que te encantó. La foto llega a tu memoria con todo su contexto: dónde, cuándo y qué te pareció."
    },
    {
      "id": "ordenador",
      "side": "left",
      "art": "portatil",
      "label": "Ordenador",
      "title": "Tu trabajo",
      "event": "Decisión: presentar a inversores el jueves",
      "action": "Enviar a INFINITO",
      "packet": "Decisión + 2 tareas",
      "memory": "Jueves: presentación a inversores · 2 tareas abiertas",
      "think": ["Quedan 2 tareas abiertas para el jueves", "La demo no se ensaya desde el lunes"],
      "output": {"to": "self", "packet": "Propuesta", "result": "Propuesta: ensayar la demo antes del jueves"},
      "explain": "Tus decisiones y tareas se recuerdan con su fecha. INFINITO detecta lo que falta y te propone el siguiente paso antes de que se te escape."
    },
    {
      "id": "jardin",
      "side": "right",
      "art": "jardin",
      "label": "Casa conectada",
      "title": "Tu jardín",
      "event": "Humedad del suelo: 18 %",
      "action": "Leer sensor",
      "packet": "Humedad 18 % · tomateras",
      "memory": "Tomateras: suelo seco (18 %)",
      "think": ["Sin lluvia prevista en tres días", "Último riego: hace cuatro días", "Tus tomateras necesitan agua"],
      "output": {"to": "self", "packet": "Regar 6 min", "result": "Riego activado · 6 min"},
      "explain": "El sensor detecta el suelo seco. INFINITO lo cruza con la previsión y con lo que sabe de tu jardín, y pone en marcha el riego. Tú ni te enteras."
    },
    {
      "id": "robot",
      "side": "right",
      "art": "robot",
      "label": "Robot de cocina",
      "title": "Tu ayudante",
      "event": "«Cocina el plato que me gustó en Tailandia»",
      "action": "Pedir la cena",
      "packet": "¿Qué plato de Tailandia?",
      "memory": null,
      "recalls": "foto",
      "think": ["Recupera la foto del Pad Thai en Bangkok", "Te encantó: lo dijiste al hacer la foto", "Hay fideos de arroz y gambas en casa"],
      "output": {"to": "self", "packet": "Receta: Pad Thai", "result": "Cocinando Pad Thai · listo en 20 min"},
      "explain": "No buscas la foto ni se la envías. El robot pregunta a la memoria común, recupera tu viaje, sabe que te encantó y se pone a cocinar."
    },
    {
      "id": "teams",
      "side": "right",
      "art": "empresa",
      "label": "INFINITO Teams",
      "title": "Tu empresa",
      "event": "Ventas: Acme pide oferta antes del viernes",
      "action": "Compartir con el equipo",
      "packet": "Acme · oferta antes del viernes",
      "memory": "Acme pide oferta antes del viernes (Ventas)",
      "think": ["Tres días después, nadie la ha preparado", "Vence mañana", "Laura llevó la última oferta de Acme"],
      "output": {"to": "self", "packet": "Aviso a Dirección", "result": "Propuesta: asignar la oferta de Acme a Laura"},
      "explain": "Lo que sabe cada departamento vive en el cerebro de la empresa. INFINITO detecta lo que se ha quedado olvidado y propone quién debe retomarlo."
    }
  ]
};

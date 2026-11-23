/* Texts and scenes of INFINITO's interactive landing (/landing).
   Everything between the first brace and the last one is strict JSON: double quotes, no trailing commas, no comments.
   tests/test_landing_v3.py reads it that way to check that every scene says what works today and what is a vision.
   status: "hoy" works today in infinitomemory.com, "piloto" is live for a pilot, "vision" is a conceptual demo.
   output.to: "apps" lights the connected apps, "self" answers back to the same scene. */
window.INFINITO_LANDING = {
  "statuses": {
    "hoy": {"label": "Funciona hoy", "short": "Hoy"},
    "piloto": {"label": "En piloto", "short": "Piloto"},
    "vision": {"label": "Visión · demostración conceptual", "short": "Visión"}
  },
  "steps": [
    {"id": "recoge", "label": "Recoge"},
    {"id": "recuerda", "label": "Recuerda"},
    {"id": "analiza", "label": "Analiza"},
    {"id": "actua", "label": "Propone y actúa"}
  ],
  "apps": ["ChatGPT", "Claude", "Grok", "Gemini", "Cursor", "Apps MCP"],
  "idle": {
    "title": "Elige una escena",
    "text": "Pulsa cualquier dispositivo y mira cómo viaja la información: llega al cerebro, queda recordada y, cuando hace falta, vuelve convertida en una propuesta o una acción."
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
      "status": "hoy",
      "statusNote": "Funciona hoy con ChatGPT, Claude, Grok, Gemini, Cursor y cualquier app compatible con MCP.",
      "packet": "«Soy Enrique»",
      "memory": "Se llama Enrique",
      "think": [],
      "output": {"to": "apps", "packet": "", "result": "Todas tus IA ya lo saben"},
      "explain": "Lo dices una vez, en cualquier IA. Las demás ya lo saben: nunca más empiezas de cero."
    },
    {
      "id": "foto",
      "side": "left",
      "art": "gafas",
      "label": "Gafas inteligentes",
      "title": "Lo que ves",
      "event": "Puesto callejero · Bangkok",
      "action": "Tomar foto",
      "status": "vision",
      "statusNote": "Guardar fotos en la memoria ya funciona hoy, por invitación. Hacerlas desde unas gafas es una demostración conceptual.",
      "packet": "Foto · Pad Thai en Bangkok",
      "packetArt": "plato",
      "memory": "Foto: el Pad Thai de Bangkok · me encantó",
      "think": [],
      "output": null,
      "explain": "Las gafas fotografían el plato que te encantó. La foto viaja a tu memoria con su contexto: dónde, cuándo y qué te pareció."
    },
    {
      "id": "ordenador",
      "side": "left",
      "art": "portatil",
      "label": "Ordenador",
      "title": "Tu trabajo",
      "event": "Decisión: presentar a inversores el jueves",
      "action": "Enviar a INFINITO",
      "status": "hoy",
      "statusNote": "Decisiones, tareas y pendientes ya se guardan hoy desde ChatGPT, Claude, Codex o Cursor. La propuesta la redacta tu IA a partir de esa memoria.",
      "packet": "Decisión + 2 tareas",
      "memory": "Jueves: presentación a inversores · 2 tareas abiertas",
      "think": ["Quedan 2 tareas abiertas para el jueves", "La demo no se ensaya desde el lunes"],
      "output": {"to": "self", "packet": "Propuesta", "result": "Propuesta: ensayar la demo antes del jueves"},
      "explain": "Tus decisiones y tareas se recuerdan con su fecha. Con ellas, INFINITO detecta lo que falta y te propone el siguiente paso."
    },
    {
      "id": "jardin",
      "side": "right",
      "art": "jardin",
      "label": "Jardín conectado",
      "title": "Tu casa",
      "event": "Humedad del suelo: 18 %",
      "action": "Leer sensor",
      "status": "vision",
      "statusNote": "Demostración conceptual: hoy INFINITO no se conecta a sensores ni a sistemas de riego.",
      "packet": "Humedad 18 % · tomateras",
      "memory": "Tomateras: suelo seco (18 %)",
      "think": ["Sin lluvia prevista en tres días", "Último riego: hace cuatro días", "Tus tomateras necesitan agua"],
      "output": {"to": "self", "packet": "Regar 6 min", "result": "Riego activado · 6 min"},
      "explain": "El sensor avisa de que el suelo está seco. INFINITO lo cruza con lo que ya sabe y pone en marcha el riego, sin que tengas que pensar en ello."
    },
    {
      "id": "robot",
      "side": "right",
      "art": "robot",
      "label": "Robot de cocina",
      "title": "Tu ayudante",
      "event": "«Cocina el plato que me gustó en Tailandia»",
      "action": "Pedir la cena",
      "status": "vision",
      "statusNote": "Demostración conceptual: hoy no hay ningún robot conectado a INFINITO.",
      "packet": "¿Qué plato de Tailandia?",
      "memory": null,
      "recalls": "foto",
      "think": ["Recupera la foto del Pad Thai en Bangkok", "Te encantó: lo dijiste al hacer la foto", "Hay fideos de arroz y gambas en casa"],
      "output": {"to": "self", "packet": "Receta: Pad Thai", "result": "Cocinando Pad Thai · listo en 20 min"},
      "explain": "No buscas la foto ni se la envías. El robot pregunta a la memoria común, recupera el contexto del viaje y se pone a cocinar."
    },
    {
      "id": "teams",
      "side": "right",
      "art": "empresa",
      "label": "INFINITO Teams",
      "title": "Tu empresa",
      "event": "Ventas: Acme pide oferta antes del viernes",
      "action": "Compartir con el equipo",
      "status": "piloto",
      "statusNote": "En piloto hoy: organizaciones, espacios cifrados por proyecto, roles y permisos por app. Detectar tareas olvidadas y proponer a quién asignarlas es visión.",
      "packet": "Acme · oferta antes del viernes",
      "memory": "Acme pide oferta antes del viernes (Ventas)",
      "think": ["Tres días después, nadie la ha preparado", "Vence mañana", "Laura llevó la última oferta de Acme"],
      "output": {"to": "self", "packet": "Aviso a Dirección", "result": "Propuesta: asignar la oferta de Acme a Laura"},
      "explain": "Lo que sabe cada departamento queda en el cerebro de la empresa. INFINITO detecta lo que se ha olvidado y propone quién debe retomarlo."
    }
  ]
};

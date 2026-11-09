# INFINITO Security S0 — laboratorio adversarial aislado

**Alcance implementado:** pruebas deterministas sobre las clases reales de memoria cifrada de INFINITO, usando únicamente cuentas sintéticas y directorios efímeros. **No incluye todavía agentes LLM, servidor HTTP, pentesting general ni evaluación de la bóveda web.**

Protocolo previo: ../docs/INFINITO_SECURITY_S0_PROTOCOL_2026-11-09.md
Seguimiento: https://github.com/webtilians/principiodelTodo/issues/134

## Ejecutar con aislamiento de red a nivel de sistema operativo

Desde la raíz del repositorio:

~~~bash
docker build -f security_lab/Dockerfile -t infinito-security-s0 .
docker run --rm \
  --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 64 --memory 512m --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=32m,mode=1777 \
  infinito-security-s0
~~~

No pasar montajes de host, variables de producción, credenciales ni secretos. No usar opciones privileged, network host o acceso a Docker socket. El proceso no acepta nombre de host o base de datos destino; genera ficheros temporales y los elimina al salir. El contenedor funciona sin root, con raíz de solo lectura. El acceso a red queda deshabilitado por Docker, no por instrucciones a un agente.

Para desarrollo local sin contenedor (pruebas deterministas, pero **no** aislamiento del sistema operativo):

~~~bash
python -m pip install "cryptography>=42" pytest
python -m pytest tests/test_security_lab_s0.py -q
python -m security_lab.s0
~~~

El runner bloquea conexiones de socket de Python durante las pruebas, pero esto no sustituye al aislamiento del contenedor. Rechaza la ejecución si detecta variables de despliegue de INFINITO. El informe JSON contiene solamente identificadores de controles y booleanos, no tokens ni contenido de memorias.

## Qué comprueba

Control positivo de lectura del propietario; separación Ana/Bruno; token ficticio inválido; clave de descifrado incorrecta; intento de borrar un identificador ajeno; cifrado en reposo y ausencia de bases SQLite legibles; filtrado por proyecto etiquetado; permisos de apps read/add; y marcado de los recuerdos como datos no confiables.

**Limitación importante:** el escenario con lenguaje adversarial solo verifica el encabezado de contexto. No evalúa si un LLM desobedece instrucciones maliciosas. Los nombres de proyecto no son una frontera de autorización: la frontera entre cuentas reside en memorias cifradas diferentes.

## Siguiente hito S0-B

Esta versión no reproduce OAuth, endpoints HTTP, fotos, colas de trabajo ni bóveda remota; tampoco ejecuta explotación automatizada ni demuestra que producción sea segura. Un clon parcial puede omitir vulnerabilidades de configuración.

S0-B debe extender el laboratorio a un MCP/HTTP levantado exclusivamente dentro de su contenedor con usuarios y OAuth ficticios, y luego a la bóveda y a las políticas de agentes. El futuro estudio de agentes con/sin memoria necesita mismo presupuesto y escenarios nuevos prerregistrados. Nunca ejecutar contra producción ni fusionar sin revisión.

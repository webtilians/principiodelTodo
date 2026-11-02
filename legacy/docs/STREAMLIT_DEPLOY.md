# 🚀 Guía de Despliegue en Streamlit Cloud

## Paso 1: Conectar el Repositorio

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Click en **"New app"**
4. Selecciona el repositorio: `webtilians/principiodelTodo`
5. Branch: `master`
6. Main file path: `app.py`

## Paso 2: Configurar Secretos (API Key)

⚠️ **IMPORTANTE**: La API Key de OpenAI NO debe estar en el código.

1. En la página de tu app desplegada, ve a **Settings** (⚙️)
2. Click en **Secrets**
3. Añade lo siguiente:

```toml
OPENAI_API_KEY = "sk-proj-tu-api-key-aqui"
```

4. Click en **Save**

## Paso 3: Configuración Avanzada (Opcional)

Si quieres configurar recursos:

```toml
[resources]
limit = "medium"  # small, medium, large
```

## ⚠️ Limitaciones en Streamlit Cloud

- **Sin GPU**: El modelo usará CPU (más lento pero funciona)
- **Memoria limitada**: 1GB en plan gratuito
- **Sin persistencia**: Los archivos JSON se reinician al redeployar

## 🔧 Archivos necesarios

El proyecto ya incluye:
- ✅ `requirements.txt` - Dependencias
- ✅ `.streamlit/config.toml` - Tema y configuración
- ✅ `models/*.pt` - Modelos entrenados (~500KB total)

## 📱 URL Final

Tu app estará en:
```
https://[tu-usuario]-principiodeltodo-app-xxxxxx.streamlit.app
```

---

## 🧪 Probar Localmente con Secretos

Para simular Streamlit Cloud localmente:

1. Crea `.streamlit/secrets.toml` (NO subir a git):
```toml
OPENAI_API_KEY = "sk-proj-tu-key"
```

2. Ejecuta normalmente:
```bash
streamlit run app.py
```

---

*Última actualización: 27/11/2025*

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Optional

# disable the default docs/redoc paths so we can serve our custom styled reference
app = FastAPI(
    title="Personality Predictor API", 
    description="Production-grade classification engine utilizing a tuned machine learning pipeline.",
    docs_url=None, 
    redoc_url=None
)

# Load the saved model pipeline
try:
    model = joblib.load('model/model_pipeline.pkl')
except FileNotFoundError:
    model = None

# Match our dataset columns exactly
class PredictionInput(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: Optional[float] = None
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time_spent_Alone": 9.0,
                "Stage_fear": "Yes",
                "Social_event_attendance": 0.0,
                "Going_outside": 0.0,
                "Drained_after_socializing": "Yes",
                "Friends_circle_size": 0.0,
                "Post_frequency": 3.0
            }
        }
    }

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Personality Predictor API is operational. Access developer documentation at /docs"
    }

@app.get("/docs", include_in_schema=False)
def custom_docs():
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personality Predictor</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:      #f0f4f8;
      --surface: #ffffff;
      --surface2:#e8edf5;
      --border:  #d1d9e6;
      --intro:   #1d4ed8;
      --extro:   #d97706;
      --text:    #0f1e3c;
      --muted:   #64748b;
      --radius:  10px;
    }

    html, body { height: 100%; overflow: hidden; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, sans-serif;
      display: grid;
      grid-template-rows: auto 1fr;
      grid-template-columns: 1fr 1fr;
      grid-template-areas: "header header" "left right";
    }

    header {
      grid-area: header;
      padding: 16px 40px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: baseline;
      gap: 14px;
    }

    header h1 {
      font-family: Georgia, 'Times New Roman', serif;
      font-size: 1.25rem;
      font-weight: normal;
      color: var(--text);
    }

    header p { color: var(--muted); font-size: 0.82rem; }

    /* LEFT PANEL */
    .left-panel {
      grid-area: left;
      background: var(--surface);
      border-right: 1px solid var(--border);
      padding: 28px 32px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }

    .section-label {
      font-size: 0.67rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 20px;
    }

    .field { margin-bottom: 20px; }

    .field-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 8px;
    }

    .field-label { font-size: 0.84rem; color: var(--text); font-weight: 500; }

    .field-value {
      font-family: 'Courier New', monospace;
      font-size: 0.9rem;
      font-variant-numeric: tabular-nums;
      color: var(--intro);
      font-weight: 600;
    }

    input[type="range"] {
      -webkit-appearance: none;
      width: 100%;
      height: 4px;
      border-radius: 2px;
      background: var(--border);
      outline: none;
      cursor: pointer;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--intro);
      border: 2px solid var(--surface);
      box-shadow: 0 1px 4px rgba(0,0,0,0.2);
      transition: transform 0.15s;
    }

    input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }

    .toggle-group { display: flex; gap: 8px; }

    .toggle-btn {
      flex: 1;
      padding: 8px 0;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: var(--bg);
      color: var(--muted);
      font-size: 0.83rem;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.15s;
    }

    .toggle-btn.active {
      background: #eff6ff;
      border-color: var(--intro);
      color: var(--intro);
      font-weight: 600;
    }

    .toggle-btn:focus-visible { outline: 2px solid var(--intro); outline-offset: 2px; }

    .divider {
      border: none;
      border-top: 1px solid var(--border);
      margin: 4px 0 20px;
    }

    .submit-btn {
      width: 100%;
      padding: 12px;
      background: var(--intro);
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 0.9rem;
      font-family: inherit;
      font-weight: 600;
      letter-spacing: 0.02em;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
      margin-top: auto;
    }

    .submit-btn:hover { opacity: 0.88; }
    .submit-btn:active { transform: scale(0.98); }
    .submit-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .submit-btn:focus-visible { outline: 2px solid var(--intro); outline-offset: 3px; }

    /* RIGHT PANEL */
    .right-panel {
      grid-area: right;
      background: var(--bg);
      padding: 28px 36px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 28px;
    }

    .idle-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      gap: 10px;
      color: var(--muted);
    }

    .idle-icon {
      width: 52px;
      height: 52px;
      border-radius: 50%;
      border: 1px solid var(--border);
      background: var(--surface);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
    }

    .idle-state p { font-size: 0.84rem; }

    .result-section { display: none; flex-direction: column; gap: 28px; }
    .result-section.visible { display: flex; animation: fadeIn 0.3s ease; }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px;
    }

    .result-eyebrow {
      font-size: 0.67rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .result-name {
      font-family: Georgia, 'Times New Roman', serif;
      font-size: clamp(2.2rem, 4vw, 3rem);
      font-weight: normal;
      line-height: 1;
      margin-bottom: 24px;
      transition: color 0.4s;
    }

    .spectrum-wrap { display: flex; flex-direction: column; gap: 7px; }

    .spectrum-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.69rem;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .spectrum-track {
      position: relative;
      height: 5px;
      border-radius: 3px;
      background: linear-gradient(to right, var(--intro), var(--extro));
    }

    .spectrum-marker {
      position: absolute;
      top: 50%;
      width: 13px;
      height: 13px;
      border-radius: 50%;
      background: var(--text);
      border: 2px solid var(--surface);
      transform: translate(-50%, -50%);
      transition: left 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
      box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }

    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    .stat-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
    }

    .stat-label {
      font-size: 0.67rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .stat-value {
      font-family: 'Courier New', monospace;
      font-size: 1.7rem;
      font-variant-numeric: tabular-nums;
      font-weight: bold;
      transition: color 0.4s;
    }

    .error-bar {
      padding: 11px 15px;
      background: #fef2f2;
      border: 1px solid #fca5a5;
      border-radius: 8px;
      color: #b91c1c;
      font-size: 0.83rem;
      display: none;
    }

    .error-bar.visible { display: block; }

    @media (prefers-reduced-motion: reduce) {
      .result-section.visible { animation: none; }
      .spectrum-marker { transition: none; }
    }
  </style>
</head>
<body>

  <header>
    <h1>Personality Predictor</h1>
    <p>Behavioral signal classifier &mdash; introvert or extrovert</p>
  </header>

  <!-- LEFT: inputs -->
  <div class="left-panel">
    <div class="section-label">Behavioral Inputs</div>

    <div class="field">
      <div class="field-header">
        <span class="field-label">Time spent alone (hrs/day)</span>
        <span class="field-value" id="val-alone">5</span>
      </div>
      <input type="range" min="0" max="11" step="1" value="5" id="alone"
             oninput="document.getElementById('val-alone').textContent = this.value">
    </div>

    <div class="field">
      <div class="field-header">
        <span class="field-label">Social event attendance</span>
        <span class="field-value" id="val-social">5</span>
      </div>
      <input type="range" min="0" max="10" step="1" value="5" id="social"
             oninput="document.getElementById('val-social').textContent = this.value">
    </div>

    <div class="field">
      <div class="field-header">
        <span class="field-label">Going outside (frequency)</span>
        <span class="field-value" id="val-outside">3</span>
      </div>
      <input type="range" min="0" max="7" step="1" value="3" id="outside"
             oninput="document.getElementById('val-outside').textContent = this.value">
    </div>

    <div class="field">
      <div class="field-header">
        <span class="field-label">Friends circle size</span>
        <span class="field-value" id="val-friends">7</span>
      </div>
      <input type="range" min="0" max="15" step="1" value="7" id="friends"
             oninput="document.getElementById('val-friends').textContent = this.value">
    </div>

    <div class="field">
      <div class="field-header">
        <span class="field-label">Post frequency (per week)</span>
        <span class="field-value" id="val-posts">5</span>
      </div>
      <input type="range" min="0" max="10" step="1" value="5" id="posts"
             oninput="document.getElementById('val-posts').textContent = this.value">
    </div>

    <hr class="divider">

    <div class="field">
      <div class="field-header" style="margin-bottom:8px">
        <span class="field-label">Stage fear</span>
      </div>
      <div class="toggle-group" id="stage-fear">
        <button class="toggle-btn" data-val="Yes" onclick="selectToggle('stage-fear',this)">Yes</button>
        <button class="toggle-btn active" data-val="No" onclick="selectToggle('stage-fear',this)">No</button>
      </div>
    </div>

    <div class="field">
      <div class="field-header" style="margin-bottom:8px">
        <span class="field-label">Drained after socializing</span>
      </div>
      <div class="toggle-group" id="drained">
        <button class="toggle-btn" data-val="Yes" onclick="selectToggle('drained',this)">Yes</button>
        <button class="toggle-btn active" data-val="No" onclick="selectToggle('drained',this)">No</button>
      </div>
    </div>

    <button class="submit-btn" id="submit-btn" onclick="predict()">Predict Personality</button>
  </div>

  <!-- RIGHT: results -->
  <div class="right-panel">

    <div class="idle-state" id="idle-state">
      <div class="idle-icon">&#9670;</div>
      <p>Set the inputs and hit Predict</p>
    </div>

    <div class="error-bar" id="error-bar"></div>

    <div class="result-section" id="result-section">
      <div class="result-card">
        <div class="result-eyebrow">Prediction</div>
        <div class="result-name" id="result-name">—</div>
        <div class="spectrum-wrap">
          <div class="spectrum-labels">
            <span>Introvert</span>
            <span>Extrovert</span>
          </div>
          <div class="spectrum-track">
            <div class="spectrum-marker" id="spectrum-marker" style="left:50%"></div>
          </div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-label">Confidence</div>
          <div class="stat-value" id="confidence-value">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Model</div>
          <div class="stat-value" style="font-size:0.88rem; color:var(--muted); padding-top:4px; font-family:inherit; font-weight:500;">Logistic Regression</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function selectToggle(groupId, btn) {
      document.querySelectorAll('#' + groupId + ' .toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }

    function getToggleVal(groupId) {
      const active = document.querySelector('#' + groupId + ' .toggle-btn.active');
      return active ? active.dataset.val : 'No';
    }

    async function predict() {
      const btn = document.getElementById('submit-btn');
      btn.disabled = true;
      btn.textContent = 'Predicting…';

      document.getElementById('result-section').classList.remove('visible');
      document.getElementById('error-bar').classList.remove('visible');
      document.getElementById('idle-state').style.display = 'none';

      const body = {
        Time_spent_Alone:          parseInt(document.getElementById('alone').value),
        Social_event_attendance:   parseInt(document.getElementById('social').value),
        Going_outside:             parseInt(document.getElementById('outside').value),
        Friends_circle_size:       parseInt(document.getElementById('friends').value),
        Post_frequency:            parseInt(document.getElementById('posts').value),
        Stage_fear:                getToggleVal('stage-fear'),
        Drained_after_socializing: getToggleVal('drained')
      };

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');

        const isIntrovert = data.prediction === 'Introvert';
        const color = isIntrovert ? 'var(--intro)' : 'var(--extro)';
        const pct   = isIntrovert ? (1 - data.confidence) * 50 : 50 + data.confidence * 50;

        document.getElementById('result-name').textContent = data.prediction;
        document.getElementById('result-name').style.color = color;
        document.getElementById('confidence-value').textContent = (data.confidence * 100).toFixed(1) + '%';
        document.getElementById('confidence-value').style.color = color;
        document.getElementById('spectrum-marker').style.left = pct + '%';
        document.getElementById('result-section').classList.add('visible');

      } catch (err) {
        const bar = document.getElementById('error-bar');
        bar.textContent = err.message;
        bar.classList.add('visible');
        document.getElementById('idle-state').style.display = 'flex';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Predict Personality';
      }
    }
  </script>

</body>
</html>""")

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model pipeline file 'model_pipeline.pkl' is missing on the server."
        )
    
    input_dict = {
        'Time_spent_Alone': [input_data.Time_spent_Alone],
        'Stage_fear': [input_data.Stage_fear],
        'Social_event_attendance': [input_data.Social_event_attendance],
        'Going_outside': [input_data.Going_outside],
        'Drained_after_socializing': [input_data.Drained_after_socializing],
        'Friends_circle_size': [input_data.Friends_circle_size],
        'Post_frequency': [input_data.Post_frequency]
    }
    
    df_input = pd.DataFrame(input_dict)
    
    try:
        prediction_encoded = model.predict(df_input)[0]
        prediction_label = "Extrovert" if prediction_encoded == 1 else "Introvert"
        
        probabilities = model.predict_proba(df_input)[0]
        confidence = float(probabilities[prediction_encoded])
        
        return {
            "prediction": prediction_label,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
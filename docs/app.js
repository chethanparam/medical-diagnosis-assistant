// Set deployed API base URL here (e.g., https://your-app.onrender.com)
const API_BASE = window.API_BASE || "http://localhost:8000";

async function fetchSchema(){
  const r = await fetch(`${API_BASE}/schema`);
  if(!r.ok) throw new Error("Schema fetch failed");
  return r.json();
}

function symptomCard(name){
  const id = `sym_${name}`;
  return `
    <div class="symptom card">
      <label for="${id}">${name.replaceAll('_',' ')}</label>
      <input id="${id}" type="range" min="0" max="3" step="1" value="0"/>
    </div>`;
}

function collectInputs(features){
  const out = {};
  for(const f of features){
    const el = document.getElementById(`sym_${f}`);
    out[f] = Number(el?.value || 0);
  }
  return out;
}

async function predict(symptoms){
  const r = await fetch(`${API_BASE}/predict`,{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ symptoms })
  });
  if(!r.ok) throw new Error("Predict failed");
  return r.json();
}

function renderResult(data){
  const box = document.getElementById("result");
  box.classList.remove("hidden");
  const proba = Object.entries(data.proba)
    .sort((a,b)=>b[1]-a[1])
    .slice(0,6)
    .map(([k,v])=>`<div>${k}</div><div>${(v*100).toFixed(1)}%</div>`)
    .join("");

  box.innerHTML = `
    <h3>Prediction: ${data.predicted}</h3>
    <p>Confidence: ${(data.confidence*100).toFixed(1)}%</p>
    <div class="proba">${proba}</div>
  `;
}

(async function init(){
  try{
    const schema = await fetchSchema();
    const grid = document.getElementById("symptoms");
    grid.innerHTML = schema.features.map(symptomCard).join("");

    document.getElementById("predictBtn").addEventListener("click", async ()=>{
      const payload = collectInputs(schema.features);
      const res = await predict(payload);
      renderResult(res);
    });
  }catch(e){
    alert("Init failed: "+ e.message);
  }
})();
import { embed, loadModel } from "./model.js";
import { preprocessGray, toGray } from "./preprocess.js";
import { deletePerson, exportAll, importAll, listPeople, requestPersistence, savePerson } from "./store.js";
import { MIN_REFERENCES, verify } from "./verify.js";

const $ = (id) => document.getElementById(id);
const status = $("status");
let weights = null;
let cohort = null;
let people = [];
const pending = [];

function setStatus(text, kind = "") {
  status.textContent = text;
  status.dataset.kind = kind;
  status.hidden = !text;
}

function createCapture(host, onChange) {
  const node = $("capture_template").content.firstElementChild.cloneNode(true);
  host.replaceChildren(node);
  const pad = node.querySelector(".pad");
  const ctx = pad.getContext("2d");
  const fileInput = node.querySelector('input[type="file"]');
  const uploadPreview = node.querySelector(".drop img");
  const processed = node.querySelector(".processed");
  const processedCanvas = processed.querySelector("canvas");
  let mode = "draw";
  let drawn = false;
  let uploaded = null;

  const resetPad = () => {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, pad.width, pad.height);
    drawn = false;
    processed.hidden = true;
    onChange();
  };
  resetPad();

  let last = null;
  const point = (event) => {
    const rect = pad.getBoundingClientRect();
    return { x: ((event.clientX - rect.left) / rect.width) * pad.width, y: ((event.clientY - rect.top) / rect.height) * pad.height };
  };
  pad.addEventListener("pointerdown", (event) => {
    pad.setPointerCapture(event.pointerId);
    last = point(event);
  });
  pad.addEventListener("pointermove", (event) => {
    if (!last) return;
    const next = point(event);
    ctx.strokeStyle = "#14213d";
    ctx.lineWidth = Math.max(2.5, 4.5 * (event.pressure || 0.5));
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(next.x, next.y);
    ctx.stroke();
    last = next;
    if (!drawn) { drawn = true; onChange(); }
  });
  const end = () => { last = null; };
  pad.addEventListener("pointerup", end);
  pad.addEventListener("pointercancel", end);
  node.querySelector(".clear").addEventListener("click", resetPad);

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.src = url;
    await image.decode();
    uploaded = image;
    uploadPreview.src = url;
    uploadPreview.hidden = false;
    processed.hidden = true;
    onChange();
  });

  node.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => {
    mode = button.dataset.mode;
    node.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    node.querySelectorAll("[data-panel]").forEach((panel) => { panel.hidden = panel.dataset.panel !== mode; });
    processed.hidden = true;
    onChange();
  }));

  return {
    hasInput: () => (mode === "draw" ? drawn : Boolean(uploaded)),
    clear() {
      resetPad();
      uploaded = null;
      fileInput.value = "";
      uploadPreview.hidden = true;
      onChange();
    },
    read() {
      let source = pad;
      if (mode === "upload") {
        const scale = Math.min(1, 1400 / Math.max(uploaded.naturalWidth, uploaded.naturalHeight));
        source = document.createElement("canvas");
        source.width = Math.round(uploaded.naturalWidth * scale);
        source.height = Math.round(uploaded.naturalHeight * scale);
        const c = source.getContext("2d");
        c.fillStyle = "#fff";
        c.fillRect(0, 0, source.width, source.height);
        c.drawImage(uploaded, 0, 0, source.width, source.height);
      }
      const { width, height } = source;
      const rgba = source.getContext("2d").getImageData(0, 0, width, height).data;
      const result = preprocessGray(toGray(rgba, width, height), width, height);
      if (!result) throw new Error("No signature found. Use dark ink on plain light paper.");
      const view = processedCanvas.getContext("2d");
      const image = view.createImageData(128, 128);
      result.edges.forEach((v, i) => {
        const shade = v ? 20 : 250;
        image.data.set([shade, shade, shade + 5, 255], i * 4);
      });
      view.putImageData(image, 0, 0);
      processed.hidden = false;
      const thumb = document.createElement("canvas");
      const ratio = Math.min(1, 320 / width);
      thumb.width = Math.round(width * ratio);
      thumb.height = Math.round(height * ratio);
      thumb.getContext("2d").drawImage(source, 0, 0, thumb.width, thumb.height);
      return { embedding: embed(result.edges, weights), thumbnail: thumb.toDataURL("image/jpeg", 0.8) };
    },
  };
}

function switchTab(name) {
  document.querySelectorAll("[data-tab]").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.tab === name)));
  for (const tab of ["verify", "enrol", "people"]) $(`tab_${tab}`).hidden = tab !== name;
}
document.querySelectorAll("[data-tab]").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));

async function refreshPeople() {
  people = await listPeople();
  $("people_count").textContent = String(people.length);
  const select = $("verify_person");
  const previous = select.value;
  select.replaceChildren(...(people.length
    ? people.map((person) => new Option(`${person.name} (${person.signatures.length} samples)`, person.id))
    : [new Option("Enrol someone first", "")]));
  if (people.some((p) => p.id === previous)) select.value = previous;
  select.disabled = !people.length;

  const list = $("people_list");
  list.replaceChildren(...people.map((person) => {
    const item = document.createElement("li");
    const header = document.createElement("div");
    header.className = "person-head";
    const name = document.createElement("strong");
    name.textContent = person.name;
    const count = document.createElement("span");
    count.textContent = `${person.signatures.length} signatures`;
    const remove = document.createElement("button");
    remove.className = "link danger";
    remove.textContent = "Remove";
    remove.addEventListener("click", async () => {
      if (!confirm(`Remove ${person.name} and their signatures from this browser?`)) return;
      await deletePerson(person.id);
      await refreshPeople();
    });
    header.append(name, count, remove);
    const strip = document.createElement("div");
    strip.className = "strip";
    person.signatures.forEach((signature) => {
      const img = document.createElement("img");
      img.src = signature.thumbnail;
      img.alt = `Signature of ${person.name}`;
      strip.append(img);
    });
    item.append(header, strip);
    return item;
  }));
  if (!people.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No one is enrolled yet.";
    list.append(empty);
  }
  updateButtons();
}

const verifyCapture = createCapture($("verify_capture"), () => updateButtons());
const enrolCapture = createCapture($("enrol_capture"), () => updateButtons());

function updateButtons() {
  const ready = Boolean(weights);
  $("verify_button").disabled = !ready || !people.length || !verifyCapture?.hasInput();
  $("enrol_add").disabled = !ready || !enrolCapture?.hasInput();
  const name = $("enrol_name").value.trim();
  $("enrol_save").disabled = pending.length < MIN_REFERENCES || !name;
  $("enrol_note").textContent = pending.length < MIN_REFERENCES
    ? `${pending.length} of ${MIN_REFERENCES} signatures added.`
    : `${pending.length} signatures ready.`;
}
$("enrol_name").addEventListener("input", updateButtons);

function renderPending() {
  $("enrol_samples").replaceChildren(...pending.map((sample, index) => {
    const item = document.createElement("li");
    const img = document.createElement("img");
    img.src = sample.thumbnail;
    img.alt = `Sample ${index + 1}`;
    const remove = document.createElement("button");
    remove.className = "link danger";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => { pending.splice(index, 1); renderPending(); updateButtons(); });
    item.append(img, remove);
    return item;
  }));
}

$("enrol_add").addEventListener("click", () => {
  try {
    pending.push(enrolCapture.read());
    enrolCapture.clear();
    renderPending();
    setStatus("");
  } catch (error) {
    setStatus(error.message, "error");
  }
  updateButtons();
});

$("enrol_save").addEventListener("click", async () => {
  const name = $("enrol_name").value.trim();
  const existing = people.find((p) => p.name.toLowerCase() === name.toLowerCase());
  const person = existing
    ? { ...existing, signatures: [...existing.signatures, ...pending] }
    : { id: crypto.randomUUID(), name, created: new Date().toISOString(), signatures: [...pending] };
  await savePerson(person);
  pending.length = 0;
  renderPending();
  $("enrol_name").value = "";
  await refreshPeople();
  setStatus(`${name} enrolled with ${person.signatures.length} signatures.`, "ok");
  void requestPersistence();
});

$("verify_button").addEventListener("click", () => {
  const box = $("verify_result");
  try {
    const claimed = people.find((p) => p.id === $("verify_person").value);
    if (!claimed) return;
    const { embedding } = verifyCapture.read();
    const result = verify(embedding, claimed, people, cohort);
    box.hidden = false;
    box.dataset.verdict = result.verified ? "match" : "mismatch";
    box.innerHTML = "";
    const stamp = document.createElement("div");
    stamp.className = "stamp";
    stamp.textContent = result.verified ? "Verified" : "Not verified";
    const reason = document.createElement("p");
    reason.textContent = result.reason;
    const meter = document.createElement("div");
    meter.className = "meter";
    meter.innerHTML = `<span style="width:${Math.round(result.match * 100)}%"></span>`;
    const numbers = document.createElement("p");
    numbers.className = "numbers";
    numbers.textContent = `Similarity ${result.score.toFixed(3)}; ${claimed.name}'s own samples agree at ${result.expected.toFixed(3)}; closest other writer ${result.closest.name ? `(${result.closest.name}) ` : ""}${result.closest.score.toFixed(3)}.`;
    box.append(stamp, reason, meter, numbers);
    setStatus("");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

$("export_button").addEventListener("click", async () => {
  const blob = new Blob([await exportAll()], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `signatures-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

$("import_file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const count = await importAll(await file.text());
    await refreshPeople();
    setStatus(`Restored ${count} people.`, "ok");
  } catch (error) {
    setStatus(error.message, "error");
  }
  event.target.value = "";
});

(async () => {
  await refreshPeople();
  try {
    const base = new URL("./", import.meta.url);
    [weights, cohort] = await Promise.all([
      loadModel(base),
      fetch(new URL("model/cohort.json", base)).then((r) => r.json()).then((data) => ({
        centre: Float32Array.from(data.centre),
        writers: data.writers.map((group) => group.map((v) => Float32Array.from(v))),
      })),
    ]);
    setStatus("");
  } catch {
    setStatus("The model could not be loaded. Refresh the page to try again.", "error");
  }
  updateButtons();
})();

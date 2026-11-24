const DB = "signature-verification";
const STORE = "people";

function open() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function run(mode, action) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const request = action(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(request?.result);
    tx.onerror = () => reject(tx.error);
  });
}

const pack = (person) => ({ ...person, signatures: person.signatures.map((s) => ({ ...s, embedding: Array.from(s.embedding) })) });
const unpack = (person) => ({ ...person, signatures: person.signatures.map((s) => ({ ...s, embedding: Float32Array.from(s.embedding) })) });

export async function listPeople() {
  const rows = await run("readonly", (store) => store.getAll());
  return (rows ?? []).map(unpack).sort((a, b) => a.name.localeCompare(b.name));
}

export const savePerson = (person) => run("readwrite", (store) => store.put(pack(person)));
export const deletePerson = (id) => run("readwrite", (store) => store.delete(id));

export async function exportAll() {
  const people = await listPeople();
  return JSON.stringify({ app: "signature-verification", version: 1, people: people.map(pack) });
}

export async function importAll(text) {
  const data = JSON.parse(text);
  if (!Array.isArray(data.people)) throw new Error("This file is not a signature backup.");
  for (const person of data.people) await savePerson(unpack(person));
  return data.people.length;
}

export async function requestPersistence() {
  try { return await navigator.storage?.persist?.(); } catch { return false; }
}

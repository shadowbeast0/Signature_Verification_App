export const MIN_REFERENCES = 3;
export const TOLERANCE = 0.05;

export function centredCosine(a, b, centre) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] - centre[i];
    const y = b[i] - centre[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na * nb) + 1e-9);
}

export function baseline(references, centre) {
  const sims = [];
  for (let i = 0; i < references.length; i += 1) {
    for (let j = i + 1; j < references.length; j += 1) sims.push(centredCosine(references[i], references[j], centre));
  }
  return sims.length ? sims.reduce((a, b) => a + b, 0) / sims.length : 1;
}

export function similarityTo(embedding, references, centre) {
  const sims = references.map((reference) => centredCosine(embedding, reference, centre));
  return sims.reduce((a, b) => a + b, 0) / sims.length;
}

export function verify(embedding, claimed, everyone, cohort) {
  const { centre } = cohort;
  const references = claimed.signatures.map((s) => s.embedding);
  const expected = baseline(references, centre);
  const score = similarityTo(embedding, references, centre);
  const enrolled = everyone
    .filter((person) => person.id !== claimed.id && person.signatures.length)
    .map((person) => ({ name: person.name, score: similarityTo(embedding, person.signatures.map((s) => s.embedding), centre) }));
  const strangers = cohort.writers.map((writer) => ({ name: null, score: similarityTo(embedding, writer, centre) }));
  const closest = [...enrolled, ...strangers].sort((a, b) => b.score - a.score)[0];
  const matchesClaim = score >= expected - TOLERANCE;
  const closerToSomeoneElse = Boolean(closest && closest.score >= score);
  const verified = matchesClaim && !closerToSomeoneElse;
  const match = Math.max(0, Math.min(1, (score - (closest?.score ?? 0)) / Math.max(1e-6, expected - (closest?.score ?? 0))));
  const other = closest?.name ? `${closest.name}'s signature` : "another writer's signature";
  let reason;
  if (verified) reason = `Consistent with ${claimed.name}'s enrolled signatures.`;
  else if (!matchesClaim) reason = `It differs from ${claimed.name}'s enrolled signatures more than they differ from each other.`;
  else reason = `It is closer to ${other} than to ${claimed.name}'s.`;
  return { verified, score, expected, match, closest, reason };
}

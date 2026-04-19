import type {
  AlignmentExample,
  FeaturesCorr,
  ModelData,
  SessionData,
} from './types';

const cache = new Map<string, Promise<unknown>>();

function load<T>(path: string): Promise<T> {
  let p = cache.get(path) as Promise<T> | undefined;
  if (!p) {
    p = fetch(`${import.meta.env.BASE_URL}data/${path}`).then((r) => {
      if (!r.ok) throw new Error(`Failed to load ${path}: ${r.status}`);
      return r.json() as Promise<T>;
    });
    cache.set(path, p);
  }
  return p;
}

export const loadModel = () => load<ModelData>('model.json');
export const loadSessions = () => load<SessionData[]>('sessions.json');
export const loadAlignmentExamples = () =>
  load<AlignmentExample[]>('alignment_examples.json');
export const loadFeaturesCorr = () => load<FeaturesCorr>('features_corr.json');

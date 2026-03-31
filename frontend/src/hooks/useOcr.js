/**
 * useOcr — hook encapsulating the full OCR + translation flow.
 *
 * Usage:
 *   const { run, state } = useOcr();
 *   await run(file, targetLang);
 */

import { useState, useCallback } from "react";
import { ocrApi, translationApi } from "../services/api";

const INITIAL = {
  status: "idle",       // idle | recognizing | translating | done | error
  persianText: "",
  translatedText: "",
  targetLang: "en",
  regions: [],
  annotatedImageUrl: null,
  error: null,
};

export function useOcr() {
  const [state, setState] = useState(INITIAL);

  const reset = useCallback(() => setState(INITIAL), []);

  const run = useCallback(async (imageFile, targetLang = "en") => {
    setState((s) => ({ ...s, status: "recognizing", error: null, targetLang }));

    try {
      // Step 1 — OCR
      const ocrResult = await ocrApi.recognize(imageFile);
      if (!ocrResult.success) throw new Error(ocrResult.error || "OCR failed");

      setState((s) => ({
        ...s,
        status: "translating",
        persianText: ocrResult.persian_text,
        regions: ocrResult.regions,
        annotatedImageUrl: ocrResult.annotated_image_url,
      }));

      // Step 2 — Translate
      const transResult = await translationApi.translate(ocrResult.persian_text, targetLang);
      if (!transResult.success) throw new Error(transResult.error || "Translation failed");

      setState((s) => ({
        ...s,
        status: "done",
        translatedText: transResult.translated,
      }));
    } catch (err) {
      setState((s) => ({
        ...s,
        status: "error",
        error: err.message || "Unknown error",
      }));
    }
  }, []);

  return { state, run, reset };
}

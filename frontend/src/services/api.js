/**
 * api.js — Axios-based service layer for all backend calls.
 * All endpoints live under /api (proxied to Flask in dev).
 */

import axios from "axios";

const http = axios.create({ baseURL: "/api" });

// ── OCR ──────────────────────────────────────────────────────────────────────
export const ocrApi = {
  /**
   * Recognise Persian text in an image file.
   * @param {File} imageFile
   * @returns {Promise<{persian_text, regions, annotated_image_url}>}
   */
  recognize: (imageFile) => {
    const form = new FormData();
    form.append("image", imageFile);
    return http.post("/ocr/recognize", form).then((r) => r.data);
  },

  /** URL of the annotated result image. */
  resultImageUrl: (jobId) => `/api/ocr/result/${jobId}`,
};

// ── Translation ───────────────────────────────────────────────────────────────
export const translationApi = {
  /**
   * Translate Persian text to a target language.
   * @param {string} text  — Persian string
   * @param {"en"|"tl"} target — target language code
   * @returns {Promise<{translated, source, target}>}
   */
  translate: (text, target = "en") =>
    http.post("/translate/", { text, target }).then((r) => r.data),

  /** List of supported target language codes. */
  languages: () => http.get("/translate/languages").then((r) => r.data),
};

// ── Health ────────────────────────────────────────────────────────────────────
export const healthApi = {
  ping: () => http.get("/health").then((r) => r.data),
};

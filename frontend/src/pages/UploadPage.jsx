/**
 * UploadPage.jsx — Main page using blueprint panel design
 */

import React, { useState } from "react";
import { ImageDropzone } from "../components/ImageDropzone";
import { LanguageSelector } from "../components/LanguageSelector";
import { ResultsPanel } from "../components/ResultsPanel";
import { StatusBadge } from "../components/StatusBadge";
import { useOcr } from "../hooks/useOcr";

export function UploadPage() {
  const { state, run, reset } = useOcr();
  const [targetLang, setTargetLang] = useState("en");
  const [preview, setPreview] = useState(null);

  const handleFile = (file) => {
    setPreview(URL.createObjectURL(file));
    run(file, targetLang);
  };

  const handleReset = () => {
    reset();
    setPreview(null);
  };

  return (
    <div className="upload-page">

      {/* ── Left column ──────────────────────────────────────── */}
      <div>
        {/* Input panel */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-number">01</span>
            <span className="panel-title">Image Input</span>
          </div>
          <div className="panel-body">
            <ImageDropzone
              onFile={handleFile}
              disabled={state.status === "recognizing" || state.status === "translating"}
            />
            {preview && (
              <div className="preview-wrap">
                <span className="field-label">Preview</span>
                <img src={preview} alt="uploaded" className="preview-img" />
              </div>
            )}
          </div>
        </div>

        {/* Translation target panel */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-number">02</span>
            <span className="panel-title">Translation Target</span>
          </div>
          <div className="panel-body">
            <LanguageSelector value={targetLang} onChange={setTargetLang} />
            {state.status !== "idle" && (
              <button className="btn-ghost" onClick={handleReset}>
                ↩ &nbsp;Start over
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── Right column ─────────────────────────────────────── */}
      <div className="results-column">
        <div className="panel">
          <div className="panel-header">
            <span className="panel-number">03</span>
            <span className="panel-title">Recognition Output</span>
            <StatusBadge status={state.status} error={state.error} inline />
          </div>
          <div className="panel-body">
            {state.status === "idle" ? (
              <div className="no-results">
                <span className="no-results-icon">🔍</span>
                Upload an image to begin
              </div>
            ) : (
              <ResultsPanel
                persianText={state.persianText}
                translatedText={state.translatedText}
                targetLang={state.targetLang}
                regions={state.regions}
                annotatedImageUrl={state.annotatedImageUrl}
                loading={state.status === "translating"}
              />
            )}
          </div>
        </div>
      </div>

    </div>
  );
}

/**
 * ResultsPanel.jsx — Displays OCR + translation results
 */

import React from "react";

const LANG_LABELS = { en: "English", tl: "Tagalog" };

export function ResultsPanel({
  persianText, translatedText, targetLang,
  regions, annotatedImageUrl, loading,
}) {
  return (
    <div className="results-panel">

      {/* Annotated image */}
      {annotatedImageUrl && (
        <div>
          <span className="field-label">Annotated Output</span>
          <img src={annotatedImageUrl} alt="annotated" className="result-img" />
        </div>
      )}

      {/* Recognised Persian */}
      <div>
        <span className="field-label">Recognised Persian Text</span>
        <div className="text-box text-box--rtl" dir="rtl" lang="fa">
          {persianText || <span className="placeholder">— waiting for result —</span>}
        </div>
        {persianText && (
          <button className="btn-copy" onClick={() => navigator.clipboard.writeText(persianText)}>
            Copy Persian
          </button>
        )}
      </div>

      {/* Translation */}
      <div>
        <span className="field-label">Translation → {LANG_LABELS[targetLang] || targetLang}</span>
        {loading ? (
          <div className="skeleton" />
        ) : (
          <div className="text-box">
            {translatedText || <span className="placeholder">— waiting for translation —</span>}
          </div>
        )}
        {translatedText && (
          <button className="btn-copy" onClick={() => navigator.clipboard.writeText(translatedText)}>
            Copy Translation
          </button>
        )}
      </div>

      {/* Per-region detections */}
      {regions.length > 0 && (
        <div>
          <span className="field-label">Detection Regions ({regions.length})</span>
          <div className="regions-table-wrap">
            <table className="regions-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Text</th>
                  <th>Confidence</th>
                  <th>Bounding Box</th>
                </tr>
              </thead>
              <tbody>
                {regions.map((r, i) => (
                  <tr key={i}>
                    <td>{String(i + 1).padStart(2, "0")}</td>
                    <td dir="rtl" lang="fa" className="region-text">{r.text}</td>
                    <td>
                      <div className="conf-cell">
                        <span>{(r.confidence * 100).toFixed(1)}%</span>
                        <div className="conf-bar">
                          <div
                            className="conf-bar-fill"
                            style={{ width: `${r.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="region-box">
                      [{r.box.join(", ")}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

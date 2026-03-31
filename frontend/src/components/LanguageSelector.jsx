/**
 * LanguageSelector.jsx
 * Picks the translation target language.
 * Adding a new language = adding one object to LANGUAGES.
 */

import React from "react";

// Extend this list as new target languages are supported by the backend
const LANGUAGES = [
  { code: "en", label: "English",  flag: "🇬🇧" },
  { code: "tl", label: "Tagalog",  flag: "🇵🇭" },
  // { code: "ar", label: "Arabic",   flag: "🇸🇦" },  // future
];

export function LanguageSelector({ value, onChange }) {
  return (
    <div className="lang-selector">
      <label className="lang-label">Translate to</label>
      <div className="lang-options">
        {LANGUAGES.map((lang) => (
          <button
            key={lang.code}
            className={["lang-btn", value === lang.code ? "lang-btn--active" : ""].join(" ")}
            onClick={() => onChange(lang.code)}
            type="button"
          >
            <span>{lang.flag}</span>
            <span>{lang.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

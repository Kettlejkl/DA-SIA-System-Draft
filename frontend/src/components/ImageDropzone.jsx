/**
 * ImageDropzone.jsx
 * Drag-and-drop / click-to-upload area using react-dropzone.
 */

import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";

const ACCEPTED = { "image/*": [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"] };

export function ImageDropzone({ onFile, disabled }) {
  const onDrop = useCallback(
    (accepted) => {
      if (accepted.length > 0) onFile(accepted[0]);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    multiple: false,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      className={[
        "dropzone",
        isDragActive ? "dropzone--active" : "",
        disabled ? "dropzone--disabled" : "",
      ].join(" ")}
    >
      <input {...getInputProps()} />

      <div className="dropzone-inner">
        <span className="dropzone-icon">🖼️</span>
        {isDragActive ? (
          <p>Drop it here …</p>
        ) : (
          <>
            <p>Drag &amp; drop a Persian text image</p>
            <p className="dropzone-sub">or click to browse</p>
            <p className="dropzone-hint">PNG · JPG · WEBP · BMP · TIFF — max 16 MB</p>
          </>
        )}
      </div>
    </div>
  );
}

import { useRef, useState } from 'react';
import { useUploadDocumentMutation } from '../../features/documents/documentsApi';
import { DocumentType } from '@pdf-rag/contracts';

export const UploadPanel = () => {
  const [uploadDocument, { isLoading, isSuccess, isError }] = useUploadDocumentMutation();
  const [file, setFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState<DocumentType>('GENERAL');
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    form.append('documentType', documentType);
    try {
      await uploadDocument(form).unwrap();
      setFile(null);
    } catch {
      // handled by isError
    }
  };

  const onFileChange = (selected?: File | null) => {
    if (!selected) return;
    setFile(selected);
  };

  const formatSize = (size: number) => {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <form className="upload-panel" onSubmit={handleSubmit}>
      <div>
        <h3>Upload a PDF</h3>
        <p>We will validate, store, and index it automatically.</p>
      </div>
      <div
        className={`upload-drop ${dragActive ? 'is-active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragActive(false);
          onFileChange(event.dataTransfer.files?.[0] ?? null);
        }}
      >
        <div className="upload-drop-info">
          <strong>{file ? file.name : 'Drop your PDF here'}</strong>
          <span>{file ? formatSize(file.size) : 'or click to browse files'}</span>
        </div>
        <button
          type="button"
          className="btn ghost"
          onClick={() => inputRef.current?.click()}
        >
          Choose file
        </button>
        <input
          ref={inputRef}
          className="upload-input"
          type="file"
          accept="application/pdf"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
      </div>

      <div className="upload-meta">
        <div className="upload-type">
          <label htmlFor="document-type">Document type</label>
          <select
            id="document-type"
            value={documentType}
            onChange={(event) => setDocumentType(event.target.value as DocumentType)}
          >
            <option value="GENERAL">General</option>
            <option value="RESEARCH">Research Paper</option>
          </select>
        </div>
        <div className="upload-actions">
          <button className="btn primary" type="submit" disabled={!file || isLoading}>
            {isLoading ? 'Uploading...' : 'Upload'}
          </button>
          {file ? (
            <button className="btn ghost" type="button" onClick={() => setFile(null)}>
              Clear
            </button>
          ) : null}
        </div>
      </div>
      <div className="upload-hint">
        Max file size 50 MB. Research mode applies section-aware chunking.
      </div>
      {isSuccess ? <div className="upload-status success">Uploaded successfully.</div> : null}
      {isError ? <div className="upload-status error">Upload failed. Try again.</div> : null}
    </form>
  );
};

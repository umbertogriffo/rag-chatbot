import React, { useCallback, useRef } from 'react';
import { DocumentInfo } from '../services/api';
import { UploadProgress } from '../hooks/useDocuments';

interface DocumentUploadProps {
  documents: DocumentInfo[];
  uploading: UploadProgress | null;
  error: string | null;
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({
  documents,
  uploading,
  error,
  onUpload,
  onDelete,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onUpload(file);
    },
    [onUpload],
  );

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex flex-col gap-3">
      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-gray-600 p-4 text-sm text-gray-400 hover:border-green-500 hover:text-green-400 cursor-pointer transition-colors"
      >
        <span className="text-2xl">📄</span>
        <span>Drag &amp; drop or click to upload</span>
        <span className="text-xs">.md .txt .pdf .html</span>
        <input
          ref={inputRef}
          type="file"
          accept=".md,.txt,.pdf,.html"
          className="hidden"
          onChange={handleChange}
        />
      </div>

      {/* Upload progress */}
      {uploading && (
        <div className="flex flex-col gap-1">
          <span className="text-xs text-gray-400 truncate">{uploading.filename}</span>
          <div className="h-1.5 w-full rounded-full bg-gray-700">
            <div
              className="h-1.5 rounded-full bg-green-500 transition-all"
              style={{ width: `${uploading.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && <p className="text-xs text-red-400">{error}</p>}

      {/* Document list */}
      {documents.length > 0 && (
        <ul className="flex flex-col gap-1">
          {documents.map((doc) => (
            <li
              key={doc.document_id}
              className="flex items-center justify-between rounded-md bg-gray-800 px-3 py-1.5 text-sm"
            >
              <span className="truncate text-gray-300" title={doc.filename}>
                {doc.filename}{' '}
                <span className="text-gray-500 text-xs">({formatBytes(doc.size)})</span>
              </span>
              <button
                onClick={() => onDelete(doc.document_id)}
                className="ml-2 text-gray-500 hover:text-red-400 transition-colors flex-shrink-0"
                title="Delete"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default DocumentUpload;

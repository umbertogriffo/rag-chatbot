import { useCallback, useEffect, useState } from 'react';
import { deleteDocument, DocumentInfo, listDocuments, uploadDocument } from '../services/api';

export interface UploadProgress {
  filename: string;
  progress: number;
}

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [uploading, setUploading] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchDocuments = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocuments(data.documents);
    } catch {
      setError('Failed to load documents');
    }
  }, []);

  useEffect(() => {
    void fetchDocuments();
  }, [fetchDocuments]);

  const upload = useCallback(
    async (file: File) => {
      setError(null);
      setUploading({ filename: file.name, progress: 0 });
      try {
        await uploadDocument(file, (pct) => {
          setUploading({ filename: file.name, progress: pct });
        });
        await fetchDocuments();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        setError(msg);
      } finally {
        setUploading(null);
      }
    },
    [fetchDocuments],
  );

  const remove = useCallback(
    async (documentId: string) => {
      setError(null);
      try {
        await deleteDocument(documentId);
        setDocuments((prev) => prev.filter((d) => d.document_id !== documentId));
      } catch {
        setError('Failed to delete document');
      }
    },
    [],
  );

  return { documents, uploading, error, upload, remove };
}

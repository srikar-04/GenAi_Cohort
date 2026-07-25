import { useReingestDocumentMutation } from '../../features/documents/documentsApi';

type Document = {
  id: string;
  title: string;
  status: 'UPLOADED' | 'PROCESSING' | 'READY' | 'FAILED';
  documentType: 'GENERAL' | 'RESEARCH';
  createdAt: string;
};

export const DocumentList = ({ documents }: { documents: Document[] }) => {
  const [reingest, { isLoading }] = useReingestDocumentMutation();

  if (documents.length === 0) {
    return <div className="empty-state">No documents yet. Upload your first PDF.</div>;
  }

  return (
    <div className="document-list">
      {documents.map((doc) => (
        <div key={doc.id} className="document-card">
          <div>
            <h4>{doc.title}</h4>
            <p>
              {doc.documentType} - {new Date(doc.createdAt).toLocaleDateString()}
            </p>
          </div>
          <div className={`status-pill status-${doc.status.toLowerCase()}`}>
            {doc.status}
          </div>
          <div className="document-actions">
            <button
              className="btn ghost"
              onClick={() => reingest(doc.id)}
              disabled={isLoading}
            >
              Re-ingest
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};
